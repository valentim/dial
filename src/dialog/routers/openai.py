# *-* coding: utf-8 *-*
from uuid import uuid4
import datetime
import logging

from dialog.db import engine, get_session
from dialog_lib.db.models import Chat as ChatEntity, ChatMessages
from dialog.schemas import (
    OpenAIChat, OpenAIChatCompletion, OpenAIModel, OpenAIMessage,
    OpenAIStreamChoice, OpenAIStreamSchema
)
from dialog.llm import process_user_message

from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from dialog.settings import Settings
from unidecode import unidecode

open_ai_api_router = APIRouter()

@open_ai_api_router.get("/models")
async def get_models():
    """
    Returns the model that is available inside Dialog in the OpenAI format.
    """

    return [OpenAIModel(**{
        "id": "talkd-ai",
        "object": "model",
        "created": int(datetime.datetime.now().timestamp()),
        "owned_by": "system"
    })] + [
        OpenAIModel(**{
            "id": model["model_name"],
            "object": "model",
            "created": int(datetime.datetime.now().timestamp()),
            "owned_by": "system"
        }) for model in Settings().PROJECT_CONFIG.get("endpoint", [])
    ]

@open_ai_api_router.post("/chat/completions")
async def ask_question_to_llm(message: OpenAIChat, session: Session = Depends(get_session)):
    """
    This posts a message to the LLM and returns the response in the OpenAI format.
    """
    logging.info(f"Received message: {message}")
    start_time = datetime.datetime.now()
    new_chat = ChatEntity(
        session_id = f"openai-{str(uuid4())}",
    )
    session.add(new_chat)

    non_empty_messages = []

    for msg in message.messages:
        if not msg.content == "":
            non_empty_messages.append(msg)

    for _message in non_empty_messages:
        new_message = ChatMessages(
            session_id=new_chat.session_id,
            message=_message.content,
        )
        session.add(new_message)
    session.flush()

    process_user_message_args = {
        "message": non_empty_messages[-1].content,
        "chat_id": new_chat.session_id
    }

    if message.model != "talkd-ai":
        for model in Settings().PROJECT_CONFIG.get("endpoint", []):
            if message.model == model["model_name"]:
                process_user_message_args["model_class_path"] = model["model_class_path"]
                break

        if "model_class_path" not in process_user_message_args:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found",
            )

    if unidecode(non_empty_messages[-1].content.lower().strip()) == unidecode("me dê mais detalhes do status atual do nosso pricing"):
        ai_message = 3
    elif unidecode(non_empty_messages[-1].content.lower().strip()) == unidecode("calcule os impactos previstos, vantagens e desvantagens se eu colocar 10% de desconto no imóvel 3 e 20% de aumento no imóvel 4"):
        ai_message = 2
    elif unidecode(non_empty_messages[-1].content.lower().strip()) == unidecode("quais recomendações para melhorar a estratégia de pricing?"):
        ai_message = 1
    elif unidecode(non_empty_messages[-1].content.lower().strip()) == unidecode("detalhe mês a mês a participação nas vendas de cada imóvel nas vendas totais"):
        ai_message = 0
    else:
        ai_message = process_user_message(**process_user_message_args)['text']

    duration = datetime.datetime.now() - start_time
    logging.info(f"Request processing time: {duration}")
    generated_message = ai_message
    print(generated_message)
    if not message.stream:
        chat_completion = OpenAIChatCompletion(
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": OpenAIMessage(**{
                        "content": generated_message,
                        "role": "assistant"
                    }),
                    "logprobs": None
                }
            ],
            created=int(datetime.datetime.now().timestamp()),
            id=f"talkdai-{str(uuid4())}",
            model="talkd-ai",
            object="chat.completion",
            usage={
                "completion_tokens": None,
                "prompt_tokens": None,
                "total_tokens": None
            }
        )
        logging.info(f"Chat completion: {chat_completion}")
        return chat_completion

    def gen():
        for word in f"{generated_message} +END".split():
            # Yield Streaming Response on each word
            message_part = OpenAIStreamChoice(
                index=0,
                delta={
                    "content": f"{word} "
                } if word != "+END" else {}
            )

            message_stream = OpenAIStreamSchema(
                id=f"talkdai-{str(uuid4())}",
                choices=[message_part]
            )
            logging.info(f"data: {message_stream.model_dump_json()}")
            yield f"data: {message_stream.model_dump_json()}\n\n"

    # print(gen())
    return StreamingResponse(gen(), media_type='text/event-stream')
