import sys
import os
import time
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from src.model.causal_mac import CausalMAC
from src.data.nuscenes_loader import get_nuscenes_dataloader
from src.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Training")


def setup_tensorboard():
    """设置 TensorBoard 日志记录器"""
    log_dir = os.path.join(project_root, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def compute_accuracy(pred_logits, targets):
    """计算准确率和 F1 分数"""
    if pred_logits.numel() == 0 or targets.numel() == 0:
        return 0.0, 0.0

    # 将logits转换为概率
    pred_probs = torch.sigmoid(pred_logits)

    # 确保形状匹配
    if pred_probs.shape != targets.shape:
        min_size = min(pred_probs.numel(), targets.numel())
        pred_probs = pred_probs.view(-1)[:min_size]
        targets = targets.view(-1)[:min_size]

    # 计算准确率
    pred_labels = (pred_probs > 0.5).float()
    correct = (pred_labels == targets).sum().item()
    accuracy = correct / targets.numel()

    # 计算 F1 分数
    true_positives = ((pred_labels == 1) & (targets == 1)).sum().item()
    false_positives = ((pred_labels == 1) & (targets == 0)).sum().item()
    false_negatives = ((pred_labels == 0) & (targets == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return accuracy, f1


def train():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优器

    # 设置 TensorBoard
    writer = setup_tensorboard()

    # 初始化模型
    model = CausalMAC(config).to(config.device)
    logger.info(f"模型初始化完成，使用设备: {config.device}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=1e-5)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    # 创建数据加载器
    train_loader = get_nuscenes_dataloader(config.data_path, config.batch_size, "train")
    val_loader = get_nuscenes_dataloader(config.data_path, config.batch_size, "val")

    logger.info(f"训练样本数: {len(train_loader.dataset)}, 批次: {len(train_loader)}")
    logger.info(f"验证样本数: {len(val_loader.dataset)}, 批次: {len(val_loader)}")

    # 训练循环
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_f1 = 0.0
        batch_count = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            batch_loss = 0.0
            batch_acc = 0.0
            batch_f1 = 0.0
            batch_samples = 0

            for item in batch:
                # 移动数据到设备
                agents = item["agents"].to(config.device, non_blocking=True)
                relations = item["relations"].to(config.device, non_blocking=True)

                # 前向传播
                pred_relations = model(agents)

                # 计算损失
                loss = model.compute_loss(pred_relations, relations)

                # 计算准确率
                with torch.no_grad():
                    acc, f1 = compute_accuracy(pred_relations, relations)
                    batch_acc += acc
                    batch_f1 += f1
                    batch_samples += 1

                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                batch_loss += loss.item()

            # 累加批次统计
            train_loss += batch_loss / len(batch)
            train_acc += batch_acc / len(batch)
            train_f1 += batch_f1 / len(batch)
            batch_count += 1
            total_samples += len(batch)

            # 每10个批次记录一次
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} | "
                    f"批次 {batch_idx}/{len(train_loader)} | "
                    f"批次损失: {batch_loss / len(batch):.4f} | "
                    f"批次准确率: {batch_acc / len(batch):.4f}"
                )

        # 计算平均训练指标
        train_loss /= batch_count
        train_acc /= batch_count
        train_f1 /= batch_count

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0.0
                batch_acc = 0.0
                batch_f1 = 0.0

                for item in batch:
                    agents = item["agents"].to(config.device, non_blocking=True)
                    relations = item["relations"].to(config.device, non_blocking=True)

                    # 前向传播
                    pred_relations = model(agents)

                    # 计算损失
                    loss = model.compute_loss(pred_relations, relations)
                    batch_loss += loss.item()

                    # 计算准确率
                    acc, f1 = compute_accuracy(pred_relations, relations)
                    batch_acc += acc
                    batch_f1 += f1

                val_loss += batch_loss / len(batch)
                val_acc += batch_acc / len(batch)
                val_f1 += batch_f1 / len(batch)
                val_batch_count += 1

        # 计算平均验证指标
        val_loss /= val_batch_count
        val_acc /= val_batch_count
        val_f1 /= val_batch_count

        # 更新学习率
        scheduler.step(val_loss)

        # 计算 epoch 时间
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        # 打印统计信息
        logger.info(f"Epoch {epoch + 1}/{config.epochs} | "
                    f"时间: {epoch_time:.2f}s | "
                    f"总时间: {total_time / 60:.1f}m | "
                    f"训练损失: {train_loss:.4f} | "
                    f"训练准确率: {train_acc:.4f} | "
                    f"验证损失: {val_loss:.4f} | "
                    f"验证准确率: {val_acc:.4f} | "
                    f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': vars(config)
            }, config.model_save_path)
            logger.info(f"保存最佳模型到 {config.model_save_path}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                os.path.dirname(config.model_save_path),
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': vars(config)
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")

    # 训练完成
    total_time = time.time() - start_time
    logger.info(f"训练完成! 总时间: {total_time / 60:.1f} 分钟")
    writer.close()

    # 评估最佳模型
    evaluate_best_model(config.model_save_path)


def evaluate_best_model(model_path):
    """评估最佳模型性能"""
    logger.info(f"\n评估最佳模型: {model_path}")

    # 加载模型
    model = CausalMAC(config).to(config.device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建测试数据加载器
    test_loader = get_nuscenes_dataloader(config.data_path, config.batch_size, "test")

    # 评估指标
    test_loss = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    test_batch_count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch_loss = 0.0
            batch_acc = 0.0
            batch_f1 = 0.0

            for item in batch:
                agents = item["agents"].to(config.device)
                relations = item["relations"].to(config.device)

                # 前向传播
                pred_relations = model(agents)

                # 计算损失
                loss = model.compute_loss(pred_relations, relations)
                batch_loss += loss.item()

                # 计算准确率
                acc, f1 = compute_accuracy(pred_relations, relations)
                batch_acc += acc
                batch_f1 += f1

            test_loss += batch_loss / len(batch)
            test_acc += batch_acc / len(batch)
            test_f1 += batch_f1 / len(batch)
            test_batch_count += 1

    # 计算平均测试指标
    test_loss /= test_batch_count
    test_acc /= test_batch_count
    test_f1 /= test_batch_count

    logger.info(f"测试结果: 损失 = {test_loss:.4f}, 准确率 = {test_acc:.4f}, F1分数 = {test_f1:.4f}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())