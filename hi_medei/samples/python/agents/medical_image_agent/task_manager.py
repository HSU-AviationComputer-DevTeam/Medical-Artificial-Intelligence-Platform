"""Medical Image Task Manager for A2A Protocol."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, AsyncIterable

import sys
import os
import traceback

# 프로젝트 루트를 파이썬 패스에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from common.server.task_manager import InMemoryTaskManager
from common.types import (
    SendTaskRequest, SendTaskResponse, SendTaskStreamingRequest, SendTaskStreamingResponse,
    Task, TaskStatus, TaskState, Artifact, Message, Part, TextPart,
    TaskStatusUpdateEvent, InternalError, JSONRPCResponse
)

from agent import SimplifiedMedicalImageAgent
from simple_vision_pipeline import simple_vision_analysis_pipeline, process_medical_image


logger = logging.getLogger(__name__)


class MedicalImageTaskManager(InMemoryTaskManager):
    """A2A 프로토콜을 지원하는 의료 영상 작업 관리자"""
    
    def __init__(self):
        super().__init__()
        logger.info("MedicalImageTaskManager initialized")
        
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        logger.info(f"새 태스크 요청: {request.params.id}")
        
        try:
            # Task 생성
            task = await self.upsert_task(request.params)
            
            # 즉시 working 상태로 업데이트
            working_status = TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role='agent',
                    parts=[TextPart(text="의료 이미지를 분석하고 있습니다...")]
                )
            )
            task = await self.update_store(task.id, working_status, [])
            
            # 백그라운드에서 실제 처리 시작
            asyncio.create_task(self._process_task_async(task.id, request.params.message))
            
            logger.info(f"태스크 처리 시작: {task.id}")
            return SendTaskResponse(id=request.id, result=task)
            
        except Exception as e:
            logger.error(f"태스크 생성 오류: {e}")
            return SendTaskResponse(
                id=request.id,
                error=InternalError(message=f"Task creation failed: {str(e)}")
            )
    
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.info(f"스트리밍 태스크 요청: {request.params.id}")
        
        try:
            # Task 생성
            task = await self.upsert_task(request.params)
            
            # SSE 설정
            sse_event_queue = await self.setup_sse_consumer(task.id)
            
            # 즉시 working 상태로 업데이트
            working_status = TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role='agent',
                    parts=[TextPart(text="의료 이미지를 분석하고 있습니다...")]
                )
            )
            await self.update_store(task.id, working_status, [])
            
            # 상태 업데이트 이벤트 큐에 추가
            working_event = TaskStatusUpdateEvent(
                id=task.id,
                status=working_status,
                final=False
            )
            await self.enqueue_events_for_sse(task.id, working_event)
            
            # 백그라운드에서 실제 처리 시작
            asyncio.create_task(self._process_task_async_streaming(task.id, request.params.message))
            
            logger.info(f"스트리밍 태스크 처리 시작: {task.id}")
            return self.dequeue_events_for_sse(request.id, task.id, sse_event_queue)
            
        except Exception as e:
            logger.error(f"스트리밍 태스크 생성 오류: {e}")
            return SendTaskStreamingResponse(
                id=request.id,
                error=InternalError(message=f"Task streaming failed: {str(e)}")
            )
    
    async def _process_task_async(self, task_id: str, message: Message):
        """백그라운드에서 실제 의료 이미지 분석 처리"""
        try:
            logger.info(f"의료 이미지 분석 시작: {task_id}")
            # 의료 이미지 분석 실행
            result = await process_medical_image(message)
            
            # 완료 상태로 업데이트
            completed_status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role='agent',
                    parts=[TextPart(text=result)]
                )
            )
            await self.update_store(task_id, completed_status, [])
            logger.info(f"의료 이미지 분석 완료: {task_id}")
            
        except Exception as e:
            logger.error(f"의료 이미지 분석 오류: {task_id}, {e}")
            # 실패 상태로 업데이트
            failed_status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(
                    role='agent',
                    parts=[TextPart(text=f"분석 중 오류가 발생했습니다: {str(e)}")]
                )
            )
            await self.update_store(task_id, failed_status, [])
    
    async def _process_task_async_streaming(self, task_id: str, message: Message):
        """백그라운드에서 실제 의료 이미지 분석 처리 (스트리밍용)"""
        try:
            logger.info(f"스트리밍 의료 이미지 분석 시작: {task_id}")
            # 의료 이미지 분석 실행
            result = await process_medical_image(message)
            
            # 완료 상태로 업데이트
            completed_status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role='agent',
                    parts=[TextPart(text=result)]
                )
            )
            await self.update_store(task_id, completed_status, [])
            
            # 완료 이벤트 큐에 추가
            completed_event = TaskStatusUpdateEvent(
                id=task_id,
                status=completed_status,
                final=True
            )
            await self.enqueue_events_for_sse(task_id, completed_event)
            logger.info(f"스트리밍 의료 이미지 분석 완료: {task_id}")
            
        except Exception as e:
            logger.error(f"스트리밍 의료 이미지 분석 오류: {task_id}, {e}")
            # 실패 상태로 업데이트
            failed_status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(
                    role='agent',
                    parts=[TextPart(text=f"분석 중 오류가 발생했습니다: {str(e)}")]
                )
            )
            await self.update_store(task_id, failed_status, [])
            
            # 실패 이벤트 큐에 추가
            failed_event = TaskStatusUpdateEvent(
                id=task_id,
                status=failed_status,
                final=True
            )
            await self.enqueue_events_for_sse(task_id, failed_event)


# 기존의 간단한 handle_a2a_request는 제거하고 A2A 프로토콜 사용
# 이전의 복잡한 기능들 제거:
# - A2ATask 클래스 (불필요)
# - 작업 큐 관리 (OpenAI API가 동기식이므로 불필요)
# - 복잡한 모달리티 분류 (OpenAI Vision이 자동 처리)
# - PACS 연동 (현재 미사용)
# - 템플릿 기반 보고서 (HyperCLOVAX가 처리) 