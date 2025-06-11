import uuid
import os
import requests
import asyncio
import base64

import mesop as me

from common.types import Message, TextPart, FilePart, FileContent
from state.host_agent_service import (
    ListConversations,
    SendMessage,
    convert_message_to_state,
)
from state.state import AppState, SettingsState, StateMessage

from .chat_bubble import chat_bubble
from .form_render import form_sent, is_form, render_form


@me.stateclass
class PageState:
    """Local Page State"""

    conversation_id: str = ''
    message_content: str = ''
    pdf_upload_alert: str = ''
    show_pdf_alert: bool = False
    show_uploader: bool = False
    image_upload_alert: str = ''
    show_image_alert: bool = False
    show_image_uploader: bool = False
    uploaded_image_name: str = ''


def on_blur(e: me.InputBlurEvent):
    """Input handler"""
    state = me.state(PageState)
    state.message_content = e.value


async def send_message(message: str, message_id: str = ''):
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 입력 시 polling 재활성화!
    settings_state = me.state(SettingsState)
    c = next(
        (
            x
            for x in await ListConversations()
            if x.conversation_id == state.conversation_id
        ),
        None,
    )
    if not c:
        print('Conversation id ', state.conversation_id, ' not found')
    request = Message(
        id=message_id,
        role='user',
        parts=[TextPart(text=message)],
        metadata={
            'conversation_id': c.conversation_id if c else '',
            'conversation_name': c.name if c else '',
        },
    )
    # Add message to state until refresh replaces it.
    state_message = convert_message_to_state(request)
    if not app_state.messages:
        app_state.messages = []
    app_state.messages.append(state_message)
    conversation = next(
        filter(
            lambda x: x.conversation_id == c.conversation_id,
            app_state.conversations,
        ),
        None,
    )
    if conversation:
        conversation.message_ids.append(state_message.message_id)
    response = await SendMessage(request)
    # 답변이 모두 완료되면 polling을 꺼줌
    if not any(session_task.task.state != "COMPLETED" for session_task in app_state.task_list):
        app_state.polling_interval = 0


async def send_message_enter(e: me.InputEnterEvent):  # pylint: disable=unused-argument
    """Send message handler - 엔터키로 전송"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 엔터 입력 시 polling 재활성화!
    state.message_content = e.value
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = ''
    await send_message(state.message_content, message_id)


async def send_message_button(e: me.ClickEvent):  # pylint: disable=unused-argument
    """Send message button handler"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    app_state.polling_interval = 1  # 버튼 클릭 시 polling 재활성화!
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = ''
    await send_message(state.message_content, message_id)


# PDF 업로드 핸들러 함수
def on_pdf_upload(e: me.UploadEvent):
    """PDF 파일 업로드 처리"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # PDF 파일인지 확인
    file_name = e.file.name
    if not file_name.lower().endswith('.pdf'):
        state.pdf_upload_alert = '오류: PDF 파일만 업로드 가능합니다'
        state.show_pdf_alert = True
        return
    
    try:
        # 현재 작업 디렉토리의 절대 경로 가져오기
        current_dir = os.path.abspath(os.path.dirname(__file__))
        # 프로젝트 루트 디렉토리 (A2A 디렉토리 위치)
        project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
        
        # 임시 디렉토리 생성 (여러 위치에 시도)
        temp_dirs = [
            os.path.join(project_root, "temp_uploads"),  # 프로젝트 루트
            os.path.join(project_root, "A2A/samples/python/temp_uploads"),  # 서버 실행 위치
            os.path.join(project_root, "A2A", "temp_uploads")  # A2A 디렉토리
        ]
        
        # 모든 가능한 위치에 디렉토리 생성 시도
        for temp_dir in temp_dirs:
            os.makedirs(temp_dir, exist_ok=True)
        
        # 각 위치에 파일 저장 시도
        saved = False
        file_content = e.file.read()
        for temp_dir in temp_dirs:
            try:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                saved = True
                print(f"파일 저장 성공: {file_path}")
                break
            except Exception as ex:
                print(f"위치에 파일 저장 실패: {temp_dir}, 오류: {ex}")
                continue
                
        if not saved:
            raise Exception("모든 임시 디렉토리에 파일 저장 실패")
        
        # PDF 인제스트
        res = requests.post(
            'http://localhost:10000/ingest_pdf',
            json={'file_path': file_path},
            timeout=30
        )
        
        if res.ok:
            app_state.last_uploaded_pdf = file_name
            state.pdf_upload_alert = f'PDF 파일 "{file_name}" 업로드 및 인제스트 성공!'
            state.show_pdf_alert = True
            if file_name not in app_state.uploaded_pdfs:
                app_state.uploaded_pdfs.append(file_name)
                # 업로드 성공 메시지를 대화에 추가 - 비동기 처리
                # asyncio.create_task는 현재 컨텍스트에서 사용할 수 없으므로 대신 메시지만 설정
                state.message_content = f"PDF 파일 '{file_name}'을 업로드했습니다. 이제 파일에 대해 질문해 보세요."
                # 다음에 사용자가 UI와 상호작용할 때 메시지가 전송됨
        else:
            state.pdf_upload_alert = f'업로드 실패: {res.text}'
            state.show_pdf_alert = True
    except Exception as ex:
        state.pdf_upload_alert = f'업로드 실패: {ex}'
        state.show_pdf_alert = True


def close_alert(e: me.ClickEvent):
    """알림 닫기"""
    state = me.state(PageState)
    state.show_pdf_alert = False


def toggle_pdf_uploader(e: me.ClickEvent):
    """Toggle PDF uploader visibility"""
    state = me.state(PageState)
    state.show_uploader = not state.show_uploader


# 이미지 업로드 핸들러 함수
def on_image_upload(e: me.UploadEvent):
    """의료 영상 파일 업로드 처리"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # 이미지 파일인지 확인
    file_name = e.file.name.lower()
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.tiff', '.tif']
    
    if not any(file_name.endswith(ext) for ext in allowed_extensions):
        state.image_upload_alert = '오류: 지원되는 의료 영상 파일만 업로드 가능합니다 (JPG, PNG, DICOM, TIFF)'
        state.show_image_alert = True
        return
    
    try:
        # 파일 내용을 base64로 인코딩
        file_content = e.file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # 파일 확장자에 따른 MIME 타입 설정
        mime_type = 'image/jpeg'
        if file_name.endswith('.png'):
            mime_type = 'image/png'
        elif file_name.endswith(('.dcm', '.dicom')):
            mime_type = 'application/dicom'
        elif file_name.endswith(('.tiff', '.tif')):
            mime_type = 'image/tiff'
        
        # 업로드 성공 알림
        state.uploaded_image_name = e.file.name
        state.image_upload_alert = f'의료 영상 "{e.file.name}" 업로드 성공! 분석을 시작합니다...'
        state.show_image_alert = True
        state.show_image_uploader = False  # 업로더 숨기기
        
        # 이미지 데이터를 앱 상태에 임시 저장
        app_state.temp_image_data[e.file.name] = {
            'base64': image_base64,
            'mime_type': mime_type,
            'filename': e.file.name,
            'size': len(file_content)
        }
        
        # 메시지 내용을 자동으로 설정
        state.message_content = f"업로드된 의료 영상을 분석해주세요: {e.file.name}"
        
    except Exception as ex:
        state.image_upload_alert = f'이미지 업로드 실패: {ex}'
        state.show_image_alert = True


async def send_image_message_auto(e: me.ClickEvent):
    """이미지가 업로드된 상태에서 자동으로 이미지 분석 메시지 전송"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # 업로드된 이미지 데이터가 있는지 확인
    if not app_state.temp_image_data:
        # 일반 텍스트 메시지 전송
        await send_message_button(e)
        return
    
    # 가장 최근 업로드된 이미지 사용
    image_data = list(app_state.temp_image_data.values())[-1]
    
    c = next(
        (
            x
            for x in await ListConversations()
            if x.conversation_id == state.conversation_id
        ),
        None,
    )
    
    if not c:
        print('Conversation id ', state.conversation_id, ' not found')
        return
    
    # 이미지 파일과 텍스트 메시지 생성
    message_id = str(uuid.uuid4())
    request = Message(
        id=message_id,
        role='user',
        parts=[
            TextPart(text=state.message_content or f"업로드된 의료 영상을 분석해주세요: {image_data['filename']}"),
            FilePart(
                file=FileContent(
                    name=image_data['filename'],
                    mimeType=image_data['mime_type'],
                    bytes=image_data['base64']
                )
            )
        ],
        metadata={
            'conversation_id': c.conversation_id,
            'conversation_name': c.name if c else '',
            'medical_image_analysis': True
        },
    )
    
    # 메시지를 상태에 추가
    state_message = convert_message_to_state(request)
    if not app_state.messages:
        app_state.messages = []
    app_state.messages.append(state_message)
    
    # 백그라운드 작업 추가
    app_state.background_tasks[message_id] = '의료 영상 분석 중...'
    app_state.polling_interval = 1  # 폴링 활성화
    
    # 메시지 전송
    response = await SendMessage(request)
    
    # 전송 후 임시 데이터 정리
    state.message_content = ''
    app_state.temp_image_data.clear()


def close_image_alert(e: me.ClickEvent):
    """이미지 업로드 알림 닫기"""
    state = me.state(PageState)
    state.show_image_alert = False


def toggle_image_uploader(e: me.ClickEvent):
    """Toggle 이미지 업로더 표시/숨김"""
    state = me.state(PageState)
    state.show_image_uploader = not state.show_image_uploader


async def quick_analyze_abnormal(e: me.ClickEvent):
    """이상 소견 퀵 분석"""
    state = me.state(PageState)
    state.message_content = "이상 소견이 있는지 확인해주세요"
    await send_image_message_auto(e)


async def quick_analyze_urgency(e: me.ClickEvent):
    """응급도 평가 퀵 분석"""
    state = me.state(PageState)
    state.message_content = "응급도를 평가해주세요"
    await send_image_message_auto(e)


async def quick_analyze_quality(e: me.ClickEvent):
    """영상 품질 평가 퀵 분석"""
    state = me.state(PageState)
    state.message_content = "영상 품질을 평가해주세요"
    await send_image_message_auto(e)


def clear_uploaded_image(e: me.ClickEvent):
    """업로드된 이미지 삭제"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # 상태 초기화
    state.uploaded_image_name = ''
    state.show_image_uploader = False
    state.show_image_alert = False
    
    # 임시 이미지 데이터 삭제
    app_state.temp_image_data.clear()


@me.component
def conversation():
    """Conversation component"""
    page_state = me.state(PageState)
    app_state = me.state(AppState)
    if 'conversation_id' in me.query_params:
        page_state.conversation_id = me.query_params['conversation_id']
        app_state.current_conversation_id = page_state.conversation_id
    
    with me.box(
        style=me.Style(
            display='flex',
            justify_content='space-between',
            flex_direction='column',
        )
    ):
        # PDF 업로드 알림 표시
        if page_state.show_pdf_alert:
            with me.box(
                style=me.Style(
                    padding=me.Padding.all(8),
                    margin=me.Margin(bottom=8),
                    background=me.theme_var('primary'),
                    color=me.theme_var('on-primary'),
                    border_radius=4,
                    display='flex',
                    justify_content='space-between',
                    align_items='center'
                )
            ):
                me.text(page_state.pdf_upload_alert)
                with me.content_button(
                    type='icon',
                    on_click=close_alert,
                    style=me.Style(color=me.theme_var('on-primary'))
                ):
                    me.icon(icon='close')
        
        # 이미지 업로드 알림 표시
        if page_state.show_image_alert:
            with me.box(
                style=me.Style(
                    padding=me.Padding.all(8),
                    margin=me.Margin(bottom=8),
                    background=me.theme_var('secondary'),
                    color=me.theme_var('on-secondary'),
                    border_radius=4,
                    display='flex',
                    justify_content='space-between',
                    align_items='center'
                )
            ):
                me.text(page_state.image_upload_alert)
                with me.content_button(
                    type='icon',
                    on_click=close_image_alert,
                    style=me.Style(color=me.theme_var('on-secondary'))
                ):
                    me.icon(icon='close')
        
        for message in app_state.messages:
            if is_form(message):
                render_form(message, app_state)
            elif form_sent(message, app_state):
                chat_bubble(
                    StateMessage(
                        message_id=message.message_id,
                        role=message.role,
                        content=[('Form submitted', 'text/plain')],
                    ),
                    message.message_id,
                )
            else:
                chat_bubble(message, message.message_id)

        # PDF 업로더
        if page_state.show_uploader:
            me.uploader(
                label="PDF 업로드",
                on_upload=on_pdf_upload
            )
        
        # 업로드된 이미지 미리보기 및 분석 UI (기존 LLM 서비스 스타일)
        if page_state.uploaded_image_name and app_state.temp_image_data:
            # 가장 최근 업로드된 이미지 데이터 가져오기
            image_data = list(app_state.temp_image_data.values())[-1]
            
            with me.box(
                style=me.Style(
                    padding=me.Padding.all(16),
                    margin=me.Margin(bottom=16),
                    background=me.theme_var('surface'),
                    border_radius=12,
                    border=me.Border.all(me.BorderSide(width=1, color=me.theme_var('outline'))),
                    box_shadow="0 2px 8px rgba(0,0,0,0.1)"
                )
            ):
                with me.box(
                    style=me.Style(
                        display='flex',
                        flex_direction='row',
                        gap=16,
                        align_items='flex-start'
                    )
                ):
                    # 이미지 미리보기 (왼쪽)
                    with me.box(
                        style=me.Style(
                            flex_grow=1,
                            max_width=400
                        )
                    ):
                        # 이미지 표시 (base64 데이터 사용)
                        with me.box(
                            style=me.Style(
                                width="100%",
                                height=200,
                                background=f"url(data:{image_data['mime_type']};base64,{image_data['base64']}) center/contain no-repeat",
                                border_radius=8,
                                border=me.Border.all(me.BorderSide(width=1, color=me.theme_var('outline-variant')))
                            )
                        ):
                            pass
                        
                        # 파일 정보
                        me.text(
                            f"📁 {image_data['filename']}",
                            style=me.Style(
                                font_size=14,
                                color=me.theme_var('on-surface-variant'),
                                margin=me.Margin(top=8)
                            )
                        )
                        me.text(
                            f"📐 크기: {image_data.get('size', 'Unknown')} bytes",
                            style=me.Style(
                                font_size=12,
                                color=me.theme_var('on-surface-variant')
                            )
                        )
                    
                    # 분석 버튼 및 옵션 (오른쪽)
                    with me.box(
                        style=me.Style(
                            display='flex',
                            flex_direction='column',
                            gap=12,
                            min_width=200
                        )
                    ):
                        # Analyze the image 버튼 (기존 LLM 서비스 스타일)
                        with me.content_button(
                            type='flat',
                            on_click=send_image_message_auto,
                            style=me.Style(
                                background=me.theme_var('primary'),
                                color=me.theme_var('on-primary'),
                                border_radius=6,
                                padding=me.Padding.all(12),
                                width="100%",
                                font_weight="500"
                            )
                        ):
                            me.text("Analyze the image")
                        
                        # 추가 옵션들
                        me.text(
                            "🔍 분석 옵션:",
                            style=me.Style(
                                font_size=14,
                                font_weight="500",
                                color=me.theme_var('on-surface')
                            )
                        )
                        
                        # 퀵 분석 버튼들
                        with me.content_button(
                            type='flat',
                            on_click=quick_analyze_abnormal,
                            style=me.Style(
                                border=me.Border.all(me.BorderSide(width=1, color=me.theme_var('outline'))),
                                border_radius=4,
                                padding=me.Padding.all(8),
                                margin=me.Margin(bottom=4),
                                width="100%"
                            )
                        ):
                            me.text("🏥 이상 소견 검사", style=me.Style(font_size=12))
                        
                        with me.content_button(
                            type='flat',
                            on_click=quick_analyze_urgency,
                            style=me.Style(
                                border=me.Border.all(me.BorderSide(width=1, color=me.theme_var('outline'))),
                                border_radius=4,
                                padding=me.Padding.all(8),
                                margin=me.Margin(bottom=4),
                                width="100%"
                            )
                        ):
                            me.text("🚨 응급도 평가", style=me.Style(font_size=12))
                        
                        with me.content_button(
                            type='flat',
                            on_click=quick_analyze_quality,
                            style=me.Style(
                                border=me.Border.all(me.BorderSide(width=1, color=me.theme_var('outline'))),
                                border_radius=4,
                                padding=me.Padding.all(8),
                                width="100%"
                            )
                        ):
                            me.text("📊 품질 평가", style=me.Style(font_size=12))
                        
                        # 이미지 삭제 버튼
                        with me.content_button(
                            type='flat',
                            on_click=clear_uploaded_image,
                            style=me.Style(
                                color=me.theme_var('error'),
                                margin=me.Margin(top=8)
                            )
                        ):
                            me.text("🗑️ 이미지 삭제", style=me.Style(font_size=12))

        # 이미지 업로더 (이미지가 없을 때만 표시)
        elif page_state.show_image_uploader:
            with me.box(
                style=me.Style(
                    padding=me.Padding.all(12),
                    margin=me.Margin(bottom=12),
                    background=me.theme_var('surface-variant'),
                    border_radius=8,
                    border=me.Border.all(me.BorderSide(width=2, color=me.theme_var('outline')))
                )
            ):
                me.text(
                    "🏥 의료 영상 업로드", 
                    style=me.Style(
                        font_weight="bold",
                        margin=me.Margin(bottom=8),
                        color=me.theme_var('primary')
                    )
                )
                me.text(
                    "지원 형식: JPG, PNG, DICOM (.dcm), TIFF",
                    style=me.Style(
                        font_size=14,
                        color=me.theme_var('on-surface-variant'),
                        margin=me.Margin(bottom=8)
                    )
                )
                me.uploader(
                    label="의료 영상 파일 선택",
                    on_upload=on_image_upload,
                    accepted_file_types=[
                        ".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".tiff", ".tif"
                    ]
                )

        with me.box(
            style=me.Style(
                display='flex',
                flex_direction='row',
                gap=5,
                align_items='center',
                min_width=500,
                width='100%',
                background=me.theme_var('surface-variant'),
                padding=me.Padding.all(8),
                border_radius=8,
            )
        ):
            me.input(
                label='How can I help you?',
                on_blur=on_blur,
                on_enter=send_message_enter,
                style=me.Style(min_width='75vw', flex_grow=1),
            )
            # 의료 영상 업로드 버튼
            with me.content_button(
                type='flat',
                on_click=toggle_image_uploader,
                style=me.Style(
                    margin=me.Margin(right=8),
                    color=me.theme_var('secondary')
                ),
            ):
                with me.tooltip(message="의료 영상 업로드 (X-Ray, CT, MRI)"):
                    me.icon(icon='medical_services')
            # PDF 업로드 버튼
            with me.content_button(
                type='flat',
                on_click=toggle_pdf_uploader,
                style=me.Style(
                    margin=me.Margin(right=8),
                    color=me.theme_var('primary')
                ),
            ):
                with me.tooltip(message="PDF 파일 업로드"):
                    me.icon(icon='description')
            # 메시지 전송 버튼 (이미지가 있으면 이미지와 함께 전송)
            with me.content_button(
                type='flat',
                on_click=send_image_message_auto,
                style=me.Style(
                    color=me.theme_var('primary')
                ),
            ):
                me.icon(icon='send')
