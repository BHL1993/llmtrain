import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR


def vali(
        args,
        accelerator,
        model,
        vali_dataset,
        vali_loader,
        criterion,
        confusion_matrix,
        global_step,
        threshold=None
):
    """
    验证函数，计算验证集的损失、指标和混淆矩阵
    """
    total_loss = []  # 存储每批次的损失值
    total_pred = []  # 存储所有预测值
    total_true = []  # 存储所有真实标签
    total_confusion_matrix = []  # 存储混淆矩阵的中间结果

    model.eval()  # 设置模型为评估模式（关闭Dropout等）
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for i, (batch_x_enc, batch_x_dec, batch_y) in enumerate(tqdm(vali_loader)):
            # 数据预处理：将数据转换为指定类型并移动到设备（GPU/TPU）
            batch_x_enc = batch_x_enc.to(torch.float64).to(accelerator.device)
            batch_y = batch_y.to(torch.int64).to(accelerator.device)

            # 混合精度计算（可选）
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, _ = model(batch_x_enc, batch_x_dec)
            else:
                outputs, _ = model(batch_x_enc, batch_x_dec)

            # 将模型输出和标签对齐到主设备进行计算
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            pred = outputs.detach()  # 断开梯度连接
            true = batch_y.detach()  # 断开梯度连接

            # 根据任务类型处理预测值（二分类或序列分类）
            if args.infer_mode == 'patch-classification':
                total_pred.extend(nn.Sigmoid()(pred).squeeze().cpu().tolist())
                pred = pred.float()  # 确保浮点类型
                true = true.float()
            elif args.infer_mode == 'sequence-classification':
                total_pred.extend(nn.Softmax(dim=1)(pred).squeeze().cpu().tolist())

            total_true.extend(true.squeeze().cpu().tolist())  # 存储真实标签

            # 计算损失
            loss = criterion(pred.squeeze(), true.squeeze())
            total_loss.append(loss.item() * len(batch_x_enc))  # 累计损失

            accelerator.print('vali pred:', pred)  # 打印预测值（调试用）

    # 在TensorBoard中记录预测值的分布
    if args.infer_mode == 'patch-classification':
        writer.add_histogram('outputs', total_pred, global_step)
        writer.add_histogram(
            'pos outputs',
            [pred for pred, true in zip(total_pred, total_true) if true],
            global_step
        )
        writer.add_histogram(
            'neg outputs',
            [pred for pred, true in zip(total_pred, total_true) if not true],
            global_step
        )
    elif args.infer_mode == 'sequence-classification':
        writer.add_histogram(
            'outputs',
            [pred[1] for pred in total_pred],
            global_step
        )
        writer.add_histogram(
            'pos outputs',
            [pred[1] for pred, true in zip(total_pred, total_true) if true],
            global_step
        )
        writer.add_histogram(
            'neg outputs',
            [pred[1] for pred, true in zip(total_pred, total_true) if not true],
            global_step
        )

    accelerator.print('vali total_pred:', total_pred)  # 打印所有预测值（调试用）
    total_loss = np.sum(total_loss) / len(vali_dataset)  # 计算平均损失

    # 计算阈值和AUC
    if isinstance(total_pred[0], list):
        threshold = sorted([pred[1] for pred in total_pred], reverse=True)[
            int(len(vali_dataset) * 0.154)
        ]
        auc = roc_auc_score(total_true, [pred[1] for pred in total_pred])
    else:
        threshold = sorted(total_pred, reverse=True)[
            int(len(vali_dataset) * 0.154)
        ]
        auc = roc_auc_score(total_true, total_pred)

    accelerator.print('vali threshold:', threshold)  # 打印阈值

    # 计算混淆矩阵
    tp, fp, fn, tn = confusion_matrix(
        torch.tensor(total_pred),
        torch.tensor(total_true),
        threshold=threshold,
        logits=False
    )

    # 计算指标（Precision, Recall, F1, Accuracy）
    total_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    total_recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) != 0 else 0
    )
    total_acc = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) != 0 else 0
    )

    accelerator.print(
        f'True pos:{tp} False pos:{fp} True neg:{tn} False neg:{fn}'
    )
    model.train()  # 恢复训练模式

    return (
        total_loss,
        total_precision,
        total_recall,
        total_f1,
        total_acc,
        auc
    )


def training_loop(dataset, configs, seed=42):
    """
    主训练循环函数，包含数据准备、模型训练和验证
    """
    global_step = 0  # 全局训练步数计数器
    set_seed(seed)  # 设置随机种子保证可复现

    # 配置分布式训练参数
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero3.json')
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=deepspeed_plugin
    )
    accelerator.print('accelerator init')  # 打印加速器初始化信息

    # 加载数据集和数据加载器
    (
        train_dataset,
        train_loader,
        vali_dataset,
        vali_loader,
        test_dataset,
        test_loader
    ) = get_dataloader(
        dataset,
        configs.batch_size,
        configs.num_workers,
        split_size=[360, 137, 200],
        seed=seed
    )

    # 保存数据集到文件（用于调试或复现）
    torch.save(train_dataset, f'MyDataset/train_dataset_seed{seed}.pth')
    torch.save(vali_dataset, f'MyDataset/vali_dataset_seed{seed}.pth')
    torch.save(test_dataset, f'MyDataset/test_dataset_seed{seed}.pth')

    # 打印数据集的类别分布
    accelerator.print(
        f'train set pos and neg ratio:'
        f'{train_dataset.label.value_counts() / len(train_dataset)}'
    )
    accelerator.print(
        f'vali set pos and neg ratio:'
        f'{vali_dataset.label.value_counts() / len(vali_dataset)}'
    )
    accelerator.print(
        f'test set pos and neg ratio:'
        f'{test_dataset.label.value_counts() / len(test_dataset)}'
    )

    # 计算类别权重（处理类别不平衡）
    tr_pos_neg_ratio = train_dataset.label.value_counts()
    balance_wt = round(tr_pos_neg_ratio[0] / tr_pos_neg_ratio[1], 2)

    # 根据任务类型设置损失函数的权重
    if configs.infer_mode == 'patch-classification':
        pos_weight = torch.tensor([balance_wt]).to(accelerator.device)
        accelerator.print(f'pos_weight: {pos_weight}')
    elif configs.infer_mode == 'sequence-classification':
        weight = torch.tensor([1, balance_wt]).to(torch.bfloat16).to(accelerator.device)
        accelerator.print(f'weight: {weight}')

    model = TimeLLMModel(configs).float()  # 初始化模型
    print('Success load model')
    print('model:', model)  # 打印模型结构

    # 生成模型保存路径
    setting = '{}_{}_sl{}_dm{}_nh{}_df{}'.format(
        configs.task_name,
        configs.llm_model_path.split('/')[-1],
        configs.seq_len,
        configs.d_model,
        configs.n_heads,
        configs.d_ff
    )

    path = os.path.join(
        './saved_models/',
        setting
    )
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()
    train_steps = len(train_loader)
    accelerator.print('train_steps len:', train_steps)  # 打印每个epoch的步数

    early_stopping = EarlyStopping(
        accelerator=accelerator,
        patience=configs.patience
    )  # 初始化早停类

    # 获取需要训练的参数
    trained_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_optim = Adam(trained_parameters, lr=configs.learning_rate)  # 初始化优化器

    # 学习率调度器（OneCycleLR）
    scheduler = OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate
    )

    # 根据任务类型选择损失函数
    if configs.infer_mode == 'patch-classification':
        if configs.loss_type == 'FocalLoss':
            criterion = FocalLoss(alpha=0.25)
        elif configs.loss_type == 'CrossEntropy':
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif configs.infer_mode == 'sequence-classification':
        if configs.loss_type == 'FocalLoss':
            criterion = MultiClassFocalLoss(
                alpha=torch.tensor([0.75, 0.25]).to(accelerator.device)
            )
        elif configs.loss_type == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss(weight=weight)

    # 将模型和数据加载器准备为分布式/加速模式
    (
        model,
        train_loader,
        vali_loader,
        test_loader,
        model_optim,
        scheduler
    ) = accelerator.prepare(
        model,
        train_loader,
        vali_loader,
        test_loader,
        model_optim,
        scheduler
    )

    for epoch in range(configs.train_epochs):
        iter_count = 0  # 当前epoch的迭代计数器
        train_loss = []  # 当前epoch的损失值
        total_pred = []  # 当前epoch的预测值
        model.train()  # 设置模型为训练模式
        epoch_time = time.time()  # 记录epoch开始时间

        for i, (batch_x_enc, batch_x_dec, batch_y) in enumerate(tqdm(train_loader)):
            accelerator.print(
                f'accelerator device: {accelerator.device}, step num: {i}'
            )
            iter_count += 1
            batch_x_enc = batch_x_enc.float().to(accelerator.device)

            # 混合精度训练
            if configs.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, scores = model(batch_x_enc, batch_x_dec)
                    total_pred.append(
                        nn.Sigmoid()(outputs).squeeze().tolist()
                    )
                    accelerator.print('outputs:', outputs)
                    accelerator.print('batch_y:', batch_y)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    train_loss.append(loss.item() * len(batch_x_enc))
            else:
                outputs, scores = model(batch_x_enc, batch_x_dec)

                # 根据任务类型处理预测值
                if configs.infer_mode == 'patch-classification':
                    total_pred.append(
                        nn.Sigmoid()(outputs).squeeze().tolist()
                    )
                elif configs.infer_mode == 'sequence-classification':
                    total_pred.append(
                        torch.nn.Softmax(dim=1)(outputs).squeeze().tolist()
                    )

                # 打印调试信息
                accelerator.print('outputs:', outputs)
                accelerator.print('batch_y:', batch_y)
                accelerator.print(
                    'output linear weight:',
                    model.output_projection.linear.weight
                )

                # 根据任务类型计算损失
                if configs.infer_mode == 'sequence-classification':
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                elif configs.infer_mode == 'patch-classification':
                    batch_y = batch_y.float()
                    loss = criterion(outputs[:, :, 0], batch_y[:, :, 0])

                train_loss.append(loss.item() * len(batch_x_enc))

            # 梯度累积（每累积一定步数后更新）
            if i // configs.accumulation_steps == 0:
                writer.add_scalar(
                    'train loss',
                    np.sum(train_loss) / ((i + 1) * configs.num_processes),
                    global_step
                )
                global_step += 1

            # 每50步打印调试信息
            if (i + 1) % 50 == 0:
                accelerator.print(...)