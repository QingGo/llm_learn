import torch
import typer
from typing import Optional

from .model import Transformer
from .prepare_data import create_translation_dataloaders
from .train import get_device, TranslationInference

def interactive_evaluation(model_path: str, device: str = None):
    """交互式评估模式，实时翻译英文到中文"""
    if device is None:
        device = get_device()
    
    print("=== 交互式翻译评估模式 ===")
    print(f"使用设备: {device}")
    print("正在加载模型...")
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        print(f"模型配置: {model_config}")
        # 创建模型
        model = Transformer(
            vocab_size=model_config.get('vocab_size', 100260),
            d_model=model_config.get('d_model', 512),
            seq_len=model_config.get('seq_len', 64),
            n_heads=model_config.get('n_heads', 8),
            d_hidden=model_config.get('d_hidden', 2048),
            stack=model_config.get('stack', 6)
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 创建数据处理器（这里需要重新创建，因为没有保存在检查点中）
        print("正在初始化数据处理器...")
        _, _, _, processor = create_translation_dataloaders(
            max_samples=100,  # 只需要少量样本来初始化处理器
            batch_size=1,
            length_percentile=0.9,
            verbose=False
        )
        
        # 创建推理器
        inference = TranslationInference(model, processor, device)
        
        print("模型加载完成！")
        print("\n使用说明:")
        print("- 输入英文句子进行翻译")
        print("- 输入 'quit' 或 'exit' 退出程序")
        print("- 输入 'help' 查看帮助")
        print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入英文句子: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("退出交互式评估模式。")
                    break
                elif user_input.lower() in ['help', 'h']:
                    print("\n帮助信息:")
                    print("- 输入英文句子进行翻译")
                    print("- 输入 'quit' 或 'exit' 退出程序")
                    print("- 输入 'help' 查看帮助")
                    continue
                elif not user_input:
                    print("请输入有效的英文句子。")
                    continue
                
                # 执行翻译
                print("翻译中...")
                translation, inference_time = inference.translate(user_input, max_length=50)
                
                # 显示结果
                print(f"\n原文 (EN): {user_input}")
                print(f"译文 (ZH): {translation}")
                print(f"推理时间: {inference_time:.3f} 秒")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n检测到 Ctrl+C，退出程序。")
                break
            except Exception as e:
                print(f"翻译过程中出现错误: {e}")
                continue
                
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保模型文件路径正确，并且已经完成训练。")
    except Exception as e:
        print(f"加载模型时出现错误: {e}")


app = typer.Typer(help="交互式翻译模型")

@app.command()
def interactive(
    model_path: str = typer.Option("./translation_checkpoints/best_model.pt", help="模型文件路径"),
    device: Optional[str] = typer.Option(None, help="指定设备 (cpu/cuda/mps)")
):
    """启动交互式翻译评估模式"""
    interactive_evaluation(model_path, device)