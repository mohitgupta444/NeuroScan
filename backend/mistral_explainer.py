"""
Mistral integration via LangChain.

Used for:
  1. Explaining CNN + U-Net brain MRI analysis results.
  2. Answering follow-up questions about the same scan.
"""

import os

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# LOAD .ENV FROM THIS FILE'S DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = "ministral-3b-2512"


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("========================================")
print("Mistral configuration")
print("========================================")
print("ENV file:", ENV_FILE)
print("ENV exists:", os.path.exists(ENV_FILE))
print("MISTRAL_API_KEY loaded:", bool(MISTRAL_API_KEY))
print("MISTRAL_MODEL:", MISTRAL_MODEL)
print("========================================")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a medical imaging assistant explaining the output of an
automated brain MRI analysis pipeline using a CNN classifier and a U-Net
segmentation model.

Rules:
- Explain findings in plain, clear language.
- Avoid unexplained medical jargon.
- Always mention that this is an AI research prototype and NOT a medical diagnosis.
- Recommend consulting a qualified radiologist or doctor for real medical decisions.
- Use only the information provided in the result.
- Do not invent findings.
- Keep the main explanation to 3-6 short sentences unless the user asks for more.
- For follow-up questions, remain consistent with the original result.
"""


# ============================================================
# CREATE MISTRAL LLM
# ============================================================

def _get_llm(temperature=0.3, max_tokens=500):

    if not MISTRAL_API_KEY:
        return None

    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ============================================================
# CONVERT RESULT JSON INTO LLM CONTEXT
# ============================================================

def _result_to_context(result_json):

    cls = result_json.get("classification", [])
    seg = result_json.get("segmentation", {})
    risk = result_json.get("risk", {})

    top = cls[0] if cls else {
        "name": "unknown",
        "prob": 0
    }

    cls_lines = "\n".join(
        f"  - {c.get('name', 'unknown')}: "
        f"{c.get('prob', 0) * 100:.1f}%"
        for c in cls
    )

    if seg.get("present"):

        seg_text = (
            "Tumor region detected.\n"
            f"  - Approximate location: {seg.get('region')}\n"
            f"  - Estimated area: {seg.get('area_mm2')} mm^2\n"
            f"  - Estimated max diameter: {seg.get('diam_mm')} mm\n"
            f"  - Segmentation mask confidence: "
            f"{seg.get('mask_confidence')}"
        )

    else:

        seg_text = (
            "No tumor region detected by the segmentation model."
        )

    context = f"""
Classification (CNN) results, most likely first:

{cls_lines}

Top predicted class:
{top.get('name', 'unknown')}
Confidence:
{top.get('prob', 0) * 100:.1f}%

Segmentation (U-Net) result:

{seg_text}

Risk band:
{risk.get('label')}

Risk explanation:
{risk.get('explanation')}
"""

    return context


# ============================================================
# PROMPTS
# ============================================================

_explain_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),

    (
        "human",
        """Here is the output of the MRI analysis pipeline:

{context}

Please explain this result to the user in simple language."""
    ),
])


_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),

    (
        "human",
        """Here is the MRI analysis result we are discussing:

{context}"""
    ),

    (
        "ai",
        "Understood, I have the result details in mind."
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        "{question}"
    ),
])


# ============================================================
# CHAT HISTORY CONVERTER
# ============================================================

def _history_to_lc_messages(chat_history):

    lc_messages = []

    for turn in chat_history:

        if turn.get("role") == "user":

            lc_messages.append(
                HumanMessage(
                    content=turn.get("content", "")
                )
            )

        elif turn.get("role") == "assistant":

            lc_messages.append(
                AIMessage(
                    content=turn.get("content", "")
                )
            )

    return lc_messages


# ============================================================
# EXPLAIN RESULT
# ============================================================

def explain_result(result_json):

    llm = _get_llm()

    if llm is None:

        return {
            "error": True,
            "message": (
                "Mistral API key not configured. "
                "Please check the MISTRAL_API_KEY in the .env file."
            ),
        }

    try:

        context = _result_to_context(result_json)

        chain = (
            _explain_prompt
            | llm
            | StrOutputParser()
        )

        content = chain.invoke({
            "context": context
        })

        return {
            "error": False,
            "message": content,
            "context_used": context
        }

    except Exception as e:

        print("========== MISTRAL EXPLAIN ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("============================================")

        return {
            "error": True,
            "message": (
                f"Mistral API request failed: {type(e).__name__}: {e}"
            )
        }


# ============================================================
# FOLLOW-UP CHAT
# ============================================================

def follow_up(result_json, chat_history, user_question):

    llm = _get_llm()

    if llm is None:

        return {
            "error": True,
            "message": (
                "Mistral API key not configured. "
                "Please check the MISTRAL_API_KEY in the .env file."
            ),
        }

    try:

        context = _result_to_context(result_json)

        lc_history = _history_to_lc_messages(
            chat_history
        )

        chain = (
            _followup_prompt
            | llm
            | StrOutputParser()
        )

        content = chain.invoke({
            "context": context,
            "chat_history": lc_history,
            "question": user_question,
        })

        return {
            "error": False,
            "message": content
        }

    except Exception as e:

        print("========== MISTRAL CHAT ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("========================================")

        return {
            "error": True,
            "message": (
                f"Mistral API request failed: {type(e).__name__}: {e}"
            )
        }