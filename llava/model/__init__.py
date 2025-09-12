from .language_model.llava_qwen2 import LlavaQwenForCausalLM, LlavaQwenConfig
from .language_model.llava_qwen2_slowfast import LlavaQwenSlowFastForCausalLM, LlavaQwenSlowFastConfig


from .language_model.llava_qwen3_2pathway import LlavaQwen2pathwayForCausalLM, LlavaQwen2pathwayConfig
# try:
#     from .language_model.llava_qwen3_2pathway import LlavaQwen2pathwayForCausalLM, LlavaQwen2pathwayConfig
#     print("✅ 成功导入 llava_qwen3_2pathway")
# except Exception as e:
#     print(f"❌ 导入 llava_qwen3_2pathway 失败: {e}")
#     import traceback
#     traceback.print_exc()  # 打印完整堆栈