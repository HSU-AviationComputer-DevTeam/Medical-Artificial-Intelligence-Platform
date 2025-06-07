"""
MCP (Model Context Protocol) 연결 설정 및 관리
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class MCPEndpoint:
    """MCP 엔드포인트 정보"""
    name: str
    url: str
    description: str
    auth_required: bool = False
    api_key: Optional[str] = None
    timeout: int = 30

class MCPConnectionManager:
    """MCP 연결 관리자"""
    
    def __init__(self):
        self.endpoints: Dict[str, MCPEndpoint] = {}
        self._load_default_endpoints()
    
    def _load_default_endpoints(self):
        """기본 MCP 엔드포인트 설정 로드"""
        default_endpoints = [
            MCPEndpoint(
                name="hospital_db",
                url="http://localhost:8080/mcp/hospital",
                description="병원 환자 데이터베이스"
            ),
            MCPEndpoint(
                name="medical_records",
                url="http://localhost:8081/mcp/records", 
                description="의료 기록 시스템"
            ),
            MCPEndpoint(
                name="drug_database",
                url="http://localhost:8082/mcp/drugs",
                description="약물 정보 데이터베이스"
            ),
            MCPEndpoint(
                name="lab_results",
                url="http://localhost:8083/mcp/lab",
                description="검사 결과 시스템"
            ),
            MCPEndpoint(
                name="imaging",
                url="http://localhost:8084/mcp/imaging",
                description="영상의학 시스템"
            )
        ]
        
        for endpoint in default_endpoints:
            self.endpoints[endpoint.name] = endpoint
    
    def add_endpoint(self, endpoint: MCPEndpoint):
        """새로운 MCP 엔드포인트 추가"""
        self.endpoints[endpoint.name] = endpoint
        logger.info(f"MCP 엔드포인트 추가: {endpoint.name} -> {endpoint.url}")
    
    def remove_endpoint(self, name: str):
        """MCP 엔드포인트 제거"""
        if name in self.endpoints:
            del self.endpoints[name]
            logger.info(f"MCP 엔드포인트 제거: {name}")
    
    def get_endpoint(self, name: str) -> Optional[MCPEndpoint]:
        """특정 엔드포인트 정보 조회"""
        return self.endpoints.get(name)
    
    def list_endpoints(self) -> List[str]:
        """사용 가능한 엔드포인트 목록 반환"""
        return list(self.endpoints.keys())
    
    async def test_connection(self, endpoint_name: str) -> Dict[str, any]:
        """MCP 엔드포인트 연결 테스트"""
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            return {"error": f"Unknown endpoint: {endpoint_name}"}
        
        try:
            # 헬스체크 엔드포인트 호출
            health_url = f"{endpoint.url}/health"
            
            headers = {"Content-Type": "application/json"}
            if endpoint.auth_required and endpoint.api_key:
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "status": "connected",
                            "endpoint": endpoint_name,
                            "url": endpoint.url,
                            "response": result
                        }
                    else:
                        return {
                            "status": "error",
                            "endpoint": endpoint_name,
                            "error": f"HTTP {response.status}: {await response.text()}"
                        }
        except Exception as e:
            return {
                "status": "error",
                "endpoint": endpoint_name,
                "error": str(e)
            }
    
    async def test_all_connections(self) -> Dict[str, any]:
        """모든 MCP 엔드포인트 연결 테스트"""
        results = {}
        
        for endpoint_name in self.endpoints.keys():
            result = await self.test_connection(endpoint_name)
            results[endpoint_name] = result
        
        return results
    
    def create_mcp_request(self, method: str, params: Dict[str, any], request_id: int = 1) -> Dict[str, any]:
        """표준 MCP JSON-RPC 요청 생성"""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id
        }
    
    async def send_mcp_request(self, endpoint_name: str, method: str, params: Dict[str, any]) -> Dict[str, any]:
        """MCP 요청 전송"""
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            return {"error": f"Unknown endpoint: {endpoint_name}"}
        
        request_data = self.create_mcp_request(method, params)
        
        try:
            headers = {"Content-Type": "application/json"}
            if endpoint.auth_required and endpoint.api_key:
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.url,
                    json=request_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": response.status,
                                "message": await response.text()
                            },
                            "id": request_data["id"]
                        }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Connection failed: {str(e)}"
                },
                "id": request_data["id"]
            }

# 전역 MCP 연결 관리자 인스턴스
mcp_manager = MCPConnectionManager() 