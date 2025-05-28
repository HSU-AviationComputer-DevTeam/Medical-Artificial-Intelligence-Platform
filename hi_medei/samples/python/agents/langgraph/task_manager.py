import asyncio
import logging
import os
import sys
from typing import Any, AsyncIterable

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hi_medei.samples.python.agents.langgraph.agent import PDFQAAgent
from hi_medei.samples.python.common.server.task_manager import InMemoryTaskManager
from hi_medei.samples.python.common.types import (
    Artifact,
    InternalError,
    InvalidParamsError,
    JSONRPCResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    Part,
)

# A2A 구현에 필요한 추가 클래스 정의
class TaskProgress:
    """작업 진행 상황을 나타내는 클래스"""
    def __init__(self, final: bool = False, resultFraction: float = 0.0, output=None):
        self.final = final
        self.resultFraction = resultFraction  
        self.output = output

class TaskCompletionOutput:
    """작업 완료 출력을 나타내는 클래스"""
    def __init__(self, content=None):
        self.content = content or []

class TaskUpdateParams:
    """작업 업데이트 매개변수 클래스"""
    def __init__(self, id: str, sessionId: str = None):
        self.id = id
        self.sessionId = sessionId

class StreamingTask:
    """스트리밍 작업 응답 클래스"""
    def __init__(self, task: Task, done: bool = False, taskId: str = None):
        self.task = task
        self.done = done
        self.taskId = taskId or task.id
from hi_medei.samples.python.common.utils.push_notification_auth import PushNotificationSenderAuth

logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    """A task manager for the PDF QA Agent"""

    def __init__(
        self,
        agent: PDFQAAgent,
        notification_sender_auth: PushNotificationSenderAuth,
    ):
        super().__init__()
        self.agent = agent
        self.notification_sender_auth = notification_sender_auth

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)

        try:
            async for item in self.agent.stream(
                query, task_send_params.sessionId
            ):
                is_task_complete = item.get('is_task_complete', False)
                require_user_input = item.get('require_user_input', False)
                content = item.get('content', 'No response')
                references = item.get('references', [])
                
                if references:
                    content_with_refs = content + "\n\n참고문헌:"
                    for i, ref in enumerate(references, 1):
                        content_with_refs += f"\n{i}. {ref}"
                    final_content = content_with_refs
                else:
                    final_content = content
                
                parts = [{'type': 'text', 'text': final_content}]
                artifact = None
                message = None
                end_stream = False

                if not is_task_complete and not require_user_input:
                    task_state = TaskState.WORKING
                    message = Message(role='agent', parts=parts)
                elif require_user_input:
                    task_state = TaskState.INPUT_REQUIRED
                    message = Message(role='agent', parts=parts)
                    end_stream = True
                else:
                    task_state = TaskState.COMPLETED
                    artifact = Artifact(parts=parts, index=0, append=False)
                    end_stream = True

                task_status = TaskStatus(state=task_state, message=message)
                latest_task = await self.update_store(
                    task_send_params.id,
                    task_status,
                    None if artifact is None else [artifact],
                )
                await self.send_task_notification(latest_task)

                if artifact:
                    task_artifact_update_event = TaskArtifactUpdateEvent(
                        id=task_send_params.id, artifact=artifact
                    )
                    await self.enqueue_events_for_sse(
                        task_send_params.id, task_artifact_update_event
                    )

                task_update_event = TaskStatusUpdateEvent(
                    id=task_send_params.id, status=task_status, final=end_stream
                )
                await self.enqueue_events_for_sse(
                    task_send_params.id, task_update_event
                )

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            await self.enqueue_events_for_sse(
                task_send_params.id,
                InternalError(
                    message=f'An error occurred while streaming the response: {e}'
                ),
            )

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles the 'send task' request."""
        if request.params.pushNotification:
            if not await self.set_push_notification_info(
                request.params.id, request.params.pushNotification
            ):
                return SendTaskResponse(
                    id=request.id,
                    error=InvalidParamsError(
                        message='Push notification URL is invalid'
                    ),
                )

        await self.upsert_task(request.params)
        task = await self.update_store(
            request.params.id, TaskStatus(state=TaskState.WORKING), None
        )
        await self.send_task_notification(task)

        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            agent_response = self.agent.invoke(
                query, task_send_params.sessionId
            )
        except Exception as e:
            logger.error(f'Error invoking agent: {e}')
            raise ValueError(f'Error invoking agent: {e}')
        return await self._process_agent_response(request, agent_response)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        try:
            await self.upsert_task(request.params)

            if request.params.pushNotification:
                if not await self.set_push_notification_info(
                    request.params.id, request.params.pushNotification
                ):
                    return JSONRPCResponse(
                        id=request.id,
                        error=InvalidParamsError(
                            message='Push notification URL is invalid'
                        ),
                    )

            task_send_params: TaskSendParams = request.params
            sse_event_queue = await self.setup_sse_consumer(
                task_send_params.id, False
            )

            asyncio.create_task(self._run_streaming_agent(request))

            return self.dequeue_events_for_sse(
                request.id, task_send_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f'Error in SSE stream: {e}')
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f'An error occurred while streaming the response: {e}'
                ),
            )

    async def _process_agent_response(
        self, request: SendTaskRequest, agent_response: dict
    ) -> SendTaskResponse:
        """Processes the agent's response and updates the task store."""
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id
        history_length = task_send_params.historyLength
        task_status = None

        content = agent_response.get('content', 'No response')
        references = agent_response.get('references', [])
        
        if references:
            content_with_refs = content + "\n\n참고문헌:"
            for i, ref in enumerate(references, 1):
                content_with_refs += f"\n{i}. {ref}"
            final_content = content_with_refs
        else:
            final_content = content
        
        parts = [{'type': 'text', 'text': final_content}]
        artifact = None
        if agent_response.get('require_user_input', False):
            task_status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(role='agent', parts=parts),
            )
        else:
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)
        task = await self.update_store(
            task_id, task_status, None if artifact is None else [artifact]
        )
        task_result = self.append_task_history(task, history_length)
        await self.send_task_notification(task)
        return SendTaskResponse(id=request.id, result=task_result)

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            raise ValueError('Only text parts are supported')
        return part.text

    async def send_task_notification(self, task: Task):
        if not await self.has_push_notification_info(task.id):
            logger.info(f'No push notification info found for task {task.id}')
            return
        push_info = await self.get_push_notification_info(task.id)

        logger.info(f'Notifying for task {task.id} => {task.status.state}')
        await self.notification_sender_auth.send_push_notification(
            push_info.url, data=task.model_dump(exclude_none=True)
        )

    async def on_resubscribe_to_task(
        self, request
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        task_id_params: TaskIdParams = request.params
        try:
            sse_event_queue = await self.setup_sse_consumer(
                task_id_params.id, True
            )
            return self.dequeue_events_for_sse(
                request.id, task_id_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f'Error while reconnecting to SSE stream: {e}')
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f'An error occurred while reconnecting to stream: {e}'
                ),
            )

    async def set_push_notification_info(
        self, task_id: str, push_notification_config: PushNotificationConfig
    ):
        # Verify the ownership of notification URL by issuing a challenge request.
        is_verified = (
            await self.notification_sender_auth.verify_push_notification_url(
                push_notification_config.url
            )
        )
        if not is_verified:
            return False

        await super().set_push_notification_info(
            task_id, push_notification_config
        )
        return True

    async def send_streaming_task(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[StreamingTask]:
        """Implements the send_streaming_task method."""
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id

        logger.info(f'Starting streaming task {task_id}')
        self.create_task_in_memory(task_id, task_send_params)

        query = self._get_input_query(task_send_params.input)
        logger.info(f'Query: {query}')

        try:
            async for update in self.agent.stream(query):
                logger.info(f'Got update from agent: {update}')
                content = update.get('content', 'No response from agent')
                references = update.get('references', [])
                
                # 응답 구성
                if references:
                    content_with_refs = content + "\n\n참고문헌:"
                    for i, ref in enumerate(references, 1):
                        content_with_refs += f"\n{i}. {ref}"
                    output_content = content_with_refs
                else:
                    output_content = content
                
                # 태스크 업데이트
                task_update_params = TaskUpdateParams(
                    id=task_id, 
                    sessionId=task_send_params.sessionId
                )
                output = TaskCompletionOutput(
                    content=[Part(type="text", text=output_content)]
                )
                
                progress = TaskProgress(
                    final=True,
                    resultFraction=1.0,
                    output=output
                )
                
                # 응답 태스크 구성
                session_task = Task(
                    id=task_id,
                    state="completed",
                    sessionId=task_send_params.sessionId,
                    progress=progress
                )
                
                # 상태 업데이트
                self.update_task_in_memory(task_id, session_task)
                
                # 스트리밍 응답 반환
                streaming_task = StreamingTask(
                    task=session_task, done=True, taskId=task_id
                )
                yield streaming_task
        
        except Exception as e:
            logger.error(f'Error processing streaming task {task_id}: {e}')
            error_response = Task(
                id=task_id,
                state="failed",
                sessionId=task_send_params.sessionId,
                error=str(e)
            )
            self.update_task_in_memory(task_id, error_response)
            yield StreamingTask(
                task=error_response, done=True, taskId=task_id
            )

    def _get_input_query(self, parts: list[Part]) -> str:
        """Extract the query from the parts."""
        for part in parts:
            if part.type == 'text':
                return part.text
        return ''
