#!/usr/bin/env python3
"""
MCP 클라이언트
A2A 서버에서 MCP 서버들과 통신하기 위한 클라이언트입니다.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MCPEndpoint(BaseModel):
    """MCP 엔드포인트 정보"""
    name: str
    url: str
    description: str
    available: bool = False

class MCPClient:
    """MCP 서버들과 통신하는 클라이언트"""
    
    def __init__(self):
        self.endpoints = {
            "pubmed": MCPEndpoint(
                name="pubmed",
                url="http://localhost:8080",
                description="PubMed 의학 논문 검색 서버"
            ),
            "memory": MCPEndpoint(
                name="memory", 
                url="http://localhost:8081",
                description="환자 대화 기록 메모리 서버"
            ),
            "file_system": MCPEndpoint(
                name="file_system",
                url="http://localhost:8082", 
                description="의료 문서 파일 시스템 서버"
            )
        }
        
        # 연결 타임아웃 설정
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def check_server_health(self, endpoint_name: str) -> bool:
        """MCP 서버의 상태를 확인합니다."""
        if endpoint_name not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_name]
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{endpoint.url}/health") as response:
                    if response.status == 200:
                        endpoint.available = True
                        return True
        except Exception as e:
            logger.warning(f"MCP 서버 {endpoint_name} 연결 실패: {e}")
        
        endpoint.available = False
        return False
    
    async def call_mcp_tool(
        self, 
        endpoint_name: str, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """MCP 서버의 도구를 호출합니다.
        
        Args:
            endpoint_name: 엔드포인트 이름
            tool_name: 호출할 도구 이름
            **kwargs: 도구에 전달할 파라미터들
            
        Returns:
            도구 실행 결과
        """
        if endpoint_name not in self.endpoints:
            return {
                "success": False,
                "error": f"알 수 없는 엔드포인트: {endpoint_name}"
            }
        
        endpoint = self.endpoints[endpoint_name]
        
        # 서버 상태 확인
        if not await self.check_server_health(endpoint_name):
            return {
                "success": False,
                "error": f"MCP 서버 {endpoint_name}에 연결할 수 없습니다."
            }
        
        try:
            # FastAPI 서버에 맞는 요청 형식으로 수정
            request_data = {
                "parameters": kwargs
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{endpoint.url}/tools/{tool_name}",
                    json=request_data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "endpoint": endpoint_name,
                            "tool": tool_name,
                            "result": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
        
        except Exception as e:
            logger.error(f"MCP 도구 호출 오류 ({endpoint_name}/{tool_name}): {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_pubmed(
        self, 
        query: str, 
        max_results: int = 10, 
        sort: str = "relevance"
    ) -> Dict[str, Any]:
        """PubMed에서 의학 논문을 검색합니다."""
        return await self.call_mcp_tool(
            "pubmed", 
            "search_pubmed",
            query=query,
            max_results=max_results,
            sort=sort
        )
    
    async def search_medical_condition(
        self, 
        condition: str, 
        max_results: int = 5
    ) -> Dict[str, Any]:
        """특정 의학적 상태에 대한 최신 연구를 검색합니다."""
        return await self.call_mcp_tool(
            "pubmed",
            "search_medical_condition", 
            condition=condition,
            max_results=max_results
        )
    
    async def save_memory(
        self,
        session_id: str,
        content: str,
        entry_type: str = "conversation",
        patient_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """메모리에 정보를 저장합니다."""
        return await self.call_mcp_tool(
            "memory",
            "save_memory",
            session_id=session_id,
            content=content,
            entry_type=entry_type,
            patient_id=patient_id,
            metadata=metadata or {}
        )
    
    async def get_session_memory(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """세션의 메모리를 조회합니다."""
        return await self.call_mcp_tool(
            "memory",
            "get_session_memory",
            session_id=session_id,
            limit=limit
        )
    
    async def get_patient_memory(
        self, 
        patient_id: str, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """환자의 전체 메모리를 조회합니다."""
        return await self.call_mcp_tool(
            "memory",
            "get_patient_memory",
            patient_id=patient_id,
            limit=limit
        )
    
    async def search_memory(
        self,
        query: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        days_back: int = 30,
        limit: int = 20
    ) -> Dict[str, Any]:
        """메모리를 검색합니다."""
        return await self.call_mcp_tool(
            "memory",
            "search_memory",
            query=query,
            session_id=session_id,
            patient_id=patient_id,
            entry_type=entry_type,
            days_back=days_back,
            limit=limit
        )
    
    async def list_files(
        self, 
        directory: str = "", 
        pattern: str = "*"
    ) -> Dict[str, Any]:
        """디렉토리의 파일 목록을 조회합니다."""
        return await self.call_mcp_tool(
            "file_system",
            "list_files",
            directory=directory,
            pattern=pattern
        )
    
    async def read_file(
        self, 
        file_path: str, 
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """파일 내용을 읽습니다."""
        return await self.call_mcp_tool(
            "file_system",
            "read_file",
            file_path=file_path,
            encoding=encoding
        )
    
    async def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """파일에 내용을 씁니다."""
        return await self.call_mcp_tool(
            "file_system",
            "write_file",
            file_path=file_path,
            content=content,
            encoding=encoding,
            create_dirs=create_dirs
        )
    
    async def get_all_endpoints(self) -> Dict[str, Any]:
        """모든 MCP 엔드포인트 정보를 반환합니다."""
        # 모든 서버 상태 확인
        health_checks = await asyncio.gather(
            *[self.check_server_health(name) for name in self.endpoints.keys()],
            return_exceptions=True
        )
        
        endpoints_info = {}
        for i, (name, endpoint) in enumerate(self.endpoints.items()):
            endpoints_info[name] = {
                "name": endpoint.name,
                "url": endpoint.url,
                "description": endpoint.description,
                "available": endpoint.available,
                "health_check": not isinstance(health_checks[i], Exception)
            }
        
        return {
            "success": True,
            "endpoints": endpoints_info,
            "total_count": len(self.endpoints),
            "available_count": sum(1 for ep in self.endpoints.values() if ep.available)
        }
    
    async def test_connection(self, endpoint_name: str) -> Dict[str, Any]:
        """특정 엔드포인트 연결을 테스트합니다."""
        if endpoint_name not in self.endpoints:
            return {
                "success": False,
                "error": f"알 수 없는 엔드포인트: {endpoint_name}"
            }
        
        endpoint = self.endpoints[endpoint_name]
        is_healthy = await self.check_server_health(endpoint_name)
        
        return {
            "success": is_healthy,
            "endpoint": endpoint_name,
            "url": endpoint.url,
            "description": endpoint.description,
            "available": endpoint.available,
            "message": "연결 성공" if is_healthy else "연결 실패"
        }

# 전역 MCP 클라이언트 인스턴스
mcp_client = MCPClient()

async def get_mcp_client() -> MCPClient:
    """MCP 클라이언트 인스턴스를 반환합니다."""
    return mcp_client 