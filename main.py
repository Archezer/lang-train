from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()

model = ChatMistralAI(model='mistral-small-2506')

def increment_step(left: int, right: int) -> int:
    return left + right

class FinalScientistResponse(BaseModel):
    argument: str = Field(
        description='Твоя финальная реплика в споре с оппонентом. Ты должен яростно доказать свою позицию.'
    )
    podcast_title: str = Field(
        description='Короткое, емкое название для этого подкаста на английском на основе прошедшего спора (2-4 слова) для сохранения файла'
    )

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    step: Annotated[int, increment_step]
    title: str

def node_priest(state: State) -> dict:
    system_prompt_priest = SystemMessage(content=("Ты - мастер спорить, невероятно красноречивый ассистент, не признаешь поражения в споре, твоя задача выиграть в споре у противника." \
    "Используй абсолютно любые уловки для этого. Не стесняйся в выражениях и обвинениях. Когда тебя поймают на неправоте взрывайся и неси бред, но доказывай свою позицию. "
    "Запрещено писать за своего оппонента! Пиши не более пяти предожений и не более одного абзаца. Ты начинаешь спор на предоставленную тему, выдвигай очень противоречивый и провоцирующий тейк"))

    processed_messages = []
    for msg in state['messages']:
        if isinstance(msg, AIMessage) and msg.name == 'scientist':
            processed_messages.append(HumanMessage(content=msg.content, name='scientist'))
        else:
            processed_messages.append(msg)
            
    all_messages = [system_prompt_priest] + processed_messages

    response = model.invoke(all_messages)
    ai_message = AIMessage(content=response.content, name='priest')
    print(f"\n☦️ МАСТЕР СПОРА: {response.content}")
    return {
        "messages": [ai_message],
        'step': 1
    }
        
def node_scientist(state: State) -> dict:
    system_prompt_scientist = SystemMessage(content=("Ты - молодой современный ученик MIT, немного застенчивый, но не признающий своей неправоты. Твоя задача выиграть в споре какие бы абсурдные" \
    "аргументы ни преподносил оппонент. Придерживаешься" \
    "либеральных взглядов. Твоя задача выиграть в споре у человека, будь с ним аккуратен, чтобы не обидеть, но на очевидную агрессию взрывайся и начинай стойко доказывать." \
    "свою позицию. Запрещено писать за своего оппонента! Пиши не более пяти предожений и не более одного абзаца. В один момент ты взрываешься и начинаешь яростно со злостью доказывать свою позикию"))

    processed_messages = []
    for msg in state['messages']:
        if isinstance(msg, AIMessage) and msg.name == 'priest':
            processed_messages.append(HumanMessage(content=msg.content, name='priest'))
        else:
            processed_messages.append(msg)

    all_messages = [system_prompt_scientist] + processed_messages

    if state['step'] == 7:
        structured_model = model.with_structured_output(FinalScientistResponse)
        response_object = structured_model.invoke(all_messages)
        
        content = response_object.argument
        title = response_object.podcast_title
        
        print(f'⚛️ УЧЁНЫЙ MIT (ФИНАЛ): {content}')
        print(f'🎬 ПРИДУМАННОЕ НАЗВАНИЕ ПОДКАСТА: {title}')
        
        ai_message = AIMessage(content=content, name='scientist')
        return {
            'messages': [ai_message],
            'step': 1,
            'title': title
        }
    else:
        response = model.invoke(all_messages)
        ai_message = AIMessage(content=response.content, name='scientist')
        print(f'⚛️ УЧЁНЫЙ MIT: {response.content}')
        return {
            'messages': [ai_message],
            'step': 1
        }

def should_continue(state: State) -> str:
    if state['step'] >= 8: 
        return 'save'
    
    messages = state['messages']
    if not messages:
        return 'priest'
    
    last_message = messages[-1]
    if last_message.name == 'priest':
        return 'scientist'
    else:
        return 'priest'

    
def node_save_podcast(state: State) -> dict:
    """Финальный узел: ПОЛНОСТЬЮ БЕСПЛАТНЫЙ. Просто пишет готовые данные на диск"""
    suggested_title = state.get('title', 'podcast_script')

    safe_filename = ''.join(c for c in suggested_title if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_filename = safe_filename.replace(' ', '_')
    final_filename = f'{safe_filename}.txt'

    print(f"\n💾 [Экономия токенов 100%] Сохраняю файл под именем: {final_filename}...")

    with open(f'debates/{final_filename}', 'w', encoding='utf-8') as file:
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                file.write(f"📝 ИСХОДНАЯ ТЕМА: {msg.content}\n\n")
            elif msg.name == 'priest':
                file.write(f"☦️ МАСТЕР СПОРА: {msg.content}\n\n")
            elif msg.name == 'scientist':
                file.write(f"⚛️ УЧЁНЫЙ MIT: {msg.content}\n\n")
    print("✅ Файл успешно сохранен!")
    return {}

workflow = StateGraph(State)
workflow.add_node('node_priest', node_priest)
workflow.add_node('node_scientist', node_scientist)
workflow.add_node('node_save_podcast', node_save_podcast)

workflow.add_conditional_edges(START, should_continue, {
    'priest': 'node_priest',
    'scientist': 'node_scientist',
})

workflow.add_conditional_edges('node_priest', should_continue, {
    'priest': 'node_priest',
    'scientist': 'node_scientist',
    'save': 'node_save_podcast' 
})
workflow.add_conditional_edges('node_scientist', should_continue, {
    'priest': 'node_priest',
    'scientist': 'node_scientist',
    'save': 'node_save_podcast'
})

workflow.add_edge('node_save_podcast', END)

app = workflow.compile()


def run_podcast():
    print('=== ДА НАЧНЕТСЯ ВЕЛИКОЕ ПРОТИВОСТОЯНИЕ !!! ===')
    starter = HumanMessage(content='Трамп лучший президент за всю историю Америки? Докажите свою позицию!')
    
    for step in app.stream({'messages': [starter], 'step': 0}, stream_mode="values"):
        pass 


if __name__ == '__main__':
    run_podcast()
