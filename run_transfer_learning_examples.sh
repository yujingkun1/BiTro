#!/bin/bash

# Cell2Gene 迁移学习运行示例脚本
# Transfer Learning Examples Script

set -e  # Exit on error

PROJECT_ROOT="/data/yujk/hovernet2feature/Cell2Gene"
cd "$PROJECT_ROOT"

echo "======================================================================="
echo "  Cell2Gene 迁移学习示例脚本"
echo "  Transfer Learning Examples"
echo "======================================================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数: 打印选项
print_option() {
    echo -e "${BLUE}[$1]${NC} $2"
}

# 函数: 打印说明
print_description() {
    echo -e "${YELLOW}说明:${NC} $1"
    echo ""
}

# 显示菜单
show_menu() {
    echo "请选择要运行的示例:"
    echo ""
    print_option "1" "快速测试 - 冻结骨干网络 (推荐) [最快, ~1-2小时]"
    print_option "2" "完整微调 - 所有层参与训练 [更慢, ~4-6小时]"
    print_option "3" "对比实验 - 迁移学习 vs 从头开始"
    print_option "4" "Leave-One-Out CV - 冻结骨干"
    print_option "5" "Leave-One-Out CV - 完整微调"
    print_option "6" "自定义配置"
    print_option "0" "退出"
    echo ""
}

# 运行训练
run_training() {
    local description="$1"
    local command="$2"
    
    echo -e "${GREEN}===== 开始训练 =====${NC}"
    echo -e "${YELLOW}描述:${NC} $description"
    echo -e "${YELLOW}命令:${NC} $command"
    echo ""
    echo "按 Enter 继续或 Ctrl+C 取消..."
    read -r
    
    echo "开始时间: $(date)"
    eval "$command"
    echo "完成时间: $(date)"
    echo ""
}

# 选项 1: 冻结骨干网络
option_1() {
    print_description "使用冻结骨干网络策略训练SpatialModel。GNN和Feature Projection层被冻结，只训练Transformer和Output Projection。这是最快的方式，适合快速验证。"
    
    run_training \
        "冻结骨干网络 + 10-Fold CV" \
        "TRANSFER_STRATEGY=frozen_backbone python spitial_model/train_transfer_learning.py"
    
    echo -e "${GREEN}结果保存在:${NC} log_normalized_transfer_frozen_backbone/"
    echo -e "${GREEN}模型保存在:${NC} checkpoints_transfer_frozen_backbone/"
}

# 选项 2: 完整微调
option_2() {
    print_description "使用完整微调策略训练SpatialModel。所有层都参与梯度更新，可获得最优性能但训练时间最长。"
    
    run_training \
        "完整微调 + 10-Fold CV" \
        "python spitial_model/train_transfer_learning.py"
    
    echo -e "${GREEN}结果保存在:${NC} log_normalized_transfer_full/"
    echo -e "${GREEN}模型保存在:${NC} checkpoints_transfer_full/"
}

# 选项 3: 对比实验
option_3() {
    echo -e "${YELLOW}此选项将运行两个训练过程 (总计 8-12 小时):${NC}"
    echo "1. 迁移学习 (冻结骨干) - 约 2 小时"
    echo "2. 无迁移学习 (从头开始) - 约 6 小时"
    echo ""
    echo "按 Enter 继续或 Ctrl+C 取消..."
    read -r
    
    echo -e "${GREEN}[第一步] 运行迁移学习 (冻结骨干)${NC}"
    TRANSFER_STRATEGY=frozen_backbone python spitial_model/train_transfer_learning.py
    
    echo ""
    echo -e "${GREEN}[第二步] 运行无迁移学习基线${NC}"
    USE_TRANSFER_LEARNING=false python spitial_model/train.py
    
    echo ""
    echo -e "${GREEN}对比分析:${NC}"
    echo "迁移学习结果: log_normalized_transfer_frozen_backbone/final_10fold_results.json"
    echo "无迁移学习基线: log_normalized/final_10fold_results.json"
    echo ""
    echo "提示: 使用 Python 脚本比较两个 JSON 文件的性能差异"
}

# 选项 4: Leave-One-Out + 冻结骨干
option_4() {
    print_description "使用 Leave-One-Out 交叉验证和冻结骨干网络策略。每次将一个样本作为测试集。"
    
    run_training \
        "Leave-One-Out CV + 冻结骨干" \
        "CV_MODE=loo TRANSFER_STRATEGY=frozen_backbone python spitial_model/train_transfer_learning.py"
    
    echo -e "${GREEN}结果保存在:${NC} log_normalized_transfer_frozen_backbone/"
}

# 选项 5: Leave-One-Out + 完整微调
option_5() {
    print_description "使用 Leave-One-Out 交叉验证和完整微调策略。需要更长时间，但可获得最优结果。"
    
    run_training \
        "Leave-One-Out CV + 完整微调" \
        "CV_MODE=loo python spitial_model/train_transfer_learning.py"
    
    echo -e "${GREEN}结果保存在:${NC} log_normalized_transfer_full/"
}

# 选项 6: 自定义配置
option_6() {
    echo -e "${YELLOW}自定义配置${NC}"
    echo ""
    echo "可用的环境变量:"
    echo "  TRANSFER_STRATEGY: full (默认) 或 frozen_backbone"
    echo "  CV_MODE: kfold (默认) 或 loo"
    echo "  LOO_HELDOUT: 逗号分隔的样本名称 (LOO模式)"
    echo ""
    
    read -p "输入自定义命令 (或按 Enter 跳过): " custom_cmd
    
    if [ -n "$custom_cmd" ]; then
        run_training "自定义配置" "$custom_cmd"
    fi
}

# 主菜单循环
main() {
    while true; do
        show_menu
        read -p "请选择 [0-6]: " choice
        
        case $choice in
            1) option_1 ;;
            2) option_2 ;;
            3) option_3 ;;
            4) option_4 ;;
            5) option_5 ;;
            6) option_6 ;;
            0) 
                echo "退出脚本"
                exit 0
                ;;
            *)
                echo -e "${YELLOW}无效选择，请重试${NC}"
                ;;
        esac
        
        echo ""
        read -p "按 Enter 继续..."
    done
}

# 检查前置条件
check_prerequisites() {
    echo "检查前置条件..."
    
    # 检查权重文件
    if [ ! -f "/data/yujk/hovernet2feature/best_bulk_static_372_optimized_model.pt" ]; then
        echo -e "${YELLOW}警告: 找不到 BulkModel 权重文件${NC}"
        echo "路径: /data/yujk/hovernet2feature/best_bulk_static_372_optimized_model.pt"
    else
        echo -e "${GREEN}✓ 找到 BulkModel 权重${NC}"
    fi
    
    # 检查数据文件
    if [ ! -d "/data/yujk/hovernet2feature/HEST/hest_data" ]; then
        echo -e "${YELLOW}警告: 找不到 HEST 数据目录${NC}"
    else
        echo -e "${GREEN}✓ 找到 HEST 数据${NC}"
    fi
    
    echo ""
}

# 启动
check_prerequisites
main

