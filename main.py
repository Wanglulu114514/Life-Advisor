import os
import random

import streamlit as st
from openai import OpenAI


DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

EXAMPLE_QUESTIONS = [
    "工作很忙但又感觉在瞎忙，我到底在干嘛？",
    "总是控制不住刷手机，时间被一点点偷走怎么办？",
    "家里催婚、同事卷薪资，我感觉被生活按在地上摩擦。",
    "想改变自己，但三分钟热度，一直坚持不下去。",
    "明知道该早睡，可就是不想结束这无聊又舒服的一天。",
]


def get_example_question_ai(api_key: str | None) -> str:
    """
    使用 DeepSeek 实时生成一个示例人生烦恼。
    如果未提供 api_key，将从本地示例列表中随机返回一个作为降级。
    """
    if not api_key:
        return random.choice(EXAMPLE_QUESTIONS)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
你是一名创意十足的“人生烦恼文案生成器”。
请每次只输出一条中文的、口语化的、简短的人生困惑或吐槽，
内容要贴近日常生活（如工作、学习、爱情、家庭、自我成长等），
长度控制在 20～40 字左右，不要加序号、解释或任何额外文字。
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "请给我一条典型的人生烦恼句子，用第一人称“我”来描述。",
        },
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.9,
        top_p=0.9,
        n=1,
        stream=False,
        max_tokens=128,
        presence_penalty=0.0,
        frequency_penalty=0.5,
        stop=None,
        logit_bias={},
        user="life-advisor-example-generator",
    )

    return response.choices[0].message.content.strip()


def get_humorous_advice(
    question: str,
    tone: str,
    temperature: float = 0.7,
    api_key: str | None = None,
) -> str:
    """
    调用 DeepSeek，生成一条幽默的人生建议。
    """
    if not api_key:
        raise ValueError("缺少 DeepSeek API Key。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = f"""
你是一位幽默、温暖、略带毒舌的人生导师。
你的任务是根据用户的困惑，给出既走心又好笑的建议。

风格要求（非常重要）：
1. 整体基调是「{tone}」，但不要太夸张。
2. 要有具体、可执行的小建议，而不是空洞鸡汤。
3. 适当用一点类比、段子、自嘲式幽默，但不能刻薄伤人。
4. 尽量用中文回答，口语一点、接地气一点。
5. 回答长度控制在 3~8 段简短分段即可，阅读轻松。
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"这是我的情况/烦恼，请给我一份幽默的人生建议：\n\n{question}",
        },
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        n=1,
        stream=False,
        max_tokens=1024,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=None,
        logit_bias={},
        user="life-advisor-ui",
    )

    return response.choices[0].message.content


def set_random_example_question():
    """按钮回调：随机切换一个示例问题。"""
    st.session_state["question_text"] = random.choice(EXAMPLE_QUESTIONS)


def main():
    st.set_page_config(
        page_title="人生建议生成器 · DeepSeek",
        page_icon="💡",
        layout="centered",
    )

    st.title("🧠 幽默人生建议小助手")
    st.markdown(
        """
欢迎来到 **DeepSeek 人生建议工作室**。

把你的烦恼交出来，我帮你：
- **一本正经地胡说八道**
- **胡说八道里带点靠谱**
- **让你笑一笑，再想一想**
        """
    )

    with st.sidebar:
        st.header("⚙️ 个性化设置")
        api_key_input = st.text_input(
            "DeepSeek API Key",
            value=st.session_state.get("deepseek_api_key", DEFAULT_API_KEY),
            type="password",
            help="可以直接在这里填写，也可以通过环境变量 DEEPSEEK_API_KEY 提前配置。",
        )
        if api_key_input:
            st.session_state["deepseek_api_key"] = api_key_input

        tone = st.selectbox(
            "建议风格",
            [
                "温柔治愈系",
                "毒舌但真诚",
                "疯癫搞笑风",
                "佛系躺平风",
                "职场冷幽默",
            ],
            index=0,
        )
        temperature = st.slider(
            "创意程度（temperature）",
            min_value=0.1,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="数值越高，回答越有想象力，也越不可预测。",
        )

    st.subheader("💬 你的问题 / 烦恼")
    default_text = "我每天都想躺平，但又对未来很焦虑，该怎么办？"
    if "question_text" not in st.session_state:
        st.session_state["question_text"] = default_text

    col1, col2 ,col3= st.columns(3)
    with col1:
        generate_btn = st.button("✨ 生成人生建议")
    with col2:
        random_examples_btn = st.button("🎲 换个示例问题")
    with col3:
        AI_examples_btn = st.button("🎲 AI 生成示例问题")

    if AI_examples_btn:
        api_key_for_example = st.session_state.get("deepseek_api_key", "").strip()
        try:
            example_q = get_example_question_ai(api_key_for_example)
        except Exception as e:
            st.warning(f"AI 生成示例失败，使用本地示例。错误信息：{e}")
            example_q = random.choice(EXAMPLE_QUESTIONS)
        st.session_state["question_text"] = example_q

    if random_examples_btn:
        example_q = random.choice(EXAMPLE_QUESTIONS)
        st.session_state["question_text"] = example_q

    st.text_area(
        "随便说说你的人生困惑，我会认真（又不太认真）地听：",
        height=150,
        key="question_text",
    )

    if generate_btn:
        api_key = st.session_state.get("deepseek_api_key", "").strip()
        question = st.session_state.get("question_text", "")

        if not api_key:
            st.error("请先在左侧输入你的 DeepSeek API Key。")
        elif not question.strip():
            st.warning("先随便说点什么吧，哪怕是一句“我也不知道说啥”。")
        else:
            with st.spinner("人生建议生成中，稍等几秒…"):
                try:
                    advice = get_humorous_advice(
                        question.strip(),
                        tone,
                        temperature,
                        api_key=api_key,
                    )
                    st.subheader("📜 给你的人生小建议")
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"调用 DeepSeek 接口出错：{e}")


if __name__ == "__main__":
    main()
