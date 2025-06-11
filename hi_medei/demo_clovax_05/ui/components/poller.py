import mesop as me

from state.host_agent_service import UpdateAppState
from state.state import AppState

# 추가: 모든 task가 COMPLETED인지 확인하는 함수
def should_continue_polling(state: AppState) -> bool:
    for session_task in state.task_list:
        if session_task.task.state != "COMPLETED":
            return True
    return False

@me.content_component
def polling_buttons():
    """Polling buttons component"""
    state = me.state(AppState)
    # polling_interval이 0(Disable)이 아니고, 미완료 task가 없으면 polling 중지
    if state.polling_interval != 0 and not should_continue_polling(state):
        state.polling_interval = 0  # 자동 Disable
    with me.box(
        style=me.Style(
            display='flex',
            justify_content='end',
        )
    ):
        me.button_toggle(
            value=[str(state.polling_interval)],
            buttons=[
                me.ButtonToggleButton(label='1s', value='1'),
                me.ButtonToggleButton(label='5s', value='5'),
                me.ButtonToggleButton(label='30s', value='30'),
                me.ButtonToggleButton(label='Disable', value='0'),
            ],
            multiple=False,
            hide_selection_indicator=True,
            disabled=False,
            on_change=on_change,
            style=me.Style(
                margin=me.Margin(bottom=20),
            ),
        )
        with me.content_button(
            type='raised',
            on_click=force_refresh,
        ):
            me.icon('refresh')
    me.slot()


def on_change(e: me.ButtonToggleChangeEvent):
    state = me.state(AppState)
    state.polling_interval = int(e.value)


async def force_refresh(e: me.ClickEvent):
    """Refresh app state event handler"""
    app_state = me.state(AppState)
    await UpdateAppState(app_state, app_state.current_conversation_id)
