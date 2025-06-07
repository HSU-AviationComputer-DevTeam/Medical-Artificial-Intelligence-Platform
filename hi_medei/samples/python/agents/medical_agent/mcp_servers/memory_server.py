#!/usr/bin/env python3
"""
메모리 FastAPI 서버
환자와의 대화 기록과 의료 정보를 저장하고 조회하는 서버입니다.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MemoryServer:
    """환자 대화 기록 및 의료 정보 저장을 위한 FastAPI 서버"""
    
    def __init__(self, port: int = 8081, db_path: str = "medical_memory.db"):
        self.port = port
        self.db_path = db_path
        self.app = FastAPI(title="Medical Memory Server")
        self._init_database()
        self._setup_routes()
    
    def _init_database(self):
        """SQLite 데이터베이스를 초기화합니다."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    entry_type TEXT DEFAULT 'conversation',
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON memory_entries(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON memory_entries(patient_id)")
            conn.commit()
    
    def _setup_routes(self):
        """FastAPI 라우트들을 설정합니다."""
        
        @self.app.get("/health")
        async def health_check():
            """헬스체크"""
            return {"status": "healthy", "service": "memory_server"}
        
        @self.app.post("/tools/save_memory")
        async def save_memory(request: Dict[str, Any]):
            """메모리에 정보를 저장합니다."""
            try:
                params = request.get("parameters", {})
                session_id = params.get("session_id", "")
                content = params.get("content", "")
                entry_type = params.get("entry_type", "conversation")
                patient_id = params.get("patient_id")
                metadata = params.get("metadata", {})
                
                if not session_id or not content:
                    return {"success": False, "error": "session_id와 content가 필요합니다."}
                
                entry_id = f"{session_id}_{int(datetime.now().timestamp() * 1000)}"
                timestamp = datetime.now().isoformat()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO memory_entries 
                        (id, patient_id, session_id, content, entry_type, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (entry_id, patient_id, session_id, content, entry_type, timestamp, json.dumps(metadata)))
                    conn.commit()
                
                return {
                    "success": True,
                    "entry_id": entry_id,
                    "message": "메모리에 저장되었습니다."
                }
            
            except Exception as e:
                logger.error(f"메모리 저장 오류: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/tools/get_session_memory")
        async def get_session_memory(request: Dict[str, Any]):
            """세션의 메모리를 조회합니다."""
            try:
                params = request.get("parameters", {})
                session_id = params.get("session_id", "")
                limit = params.get("limit", 50)
                
                if not session_id:
                    return {"success": False, "error": "session_id가 필요합니다."}
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM memory_entries 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                    
                    entries = []
                    for row in cursor.fetchall():
                        entries.append({
                            "id": row["id"],
                            "patient_id": row["patient_id"],
                            "session_id": row["session_id"],
                            "content": row["content"],
                            "entry_type": row["entry_type"],
                            "timestamp": row["timestamp"],
                            "metadata": json.loads(row["metadata"] or "{}")
                        })
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "entries": entries,
                    "total_count": len(entries)
                }
            
            except Exception as e:
                logger.error(f"세션 메모리 조회 오류: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/tools/get_patient_memory")
        async def get_patient_memory(request: Dict[str, Any]):
            """환자의 전체 메모리를 조회합니다."""
            try:
                params = request.get("parameters", {})
                patient_id = params.get("patient_id", "")
                limit = params.get("limit", 50)
                
                if not patient_id:
                    return {"success": False, "error": "patient_id가 필요합니다."}
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM memory_entries 
                        WHERE patient_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (patient_id, limit))
                    
                    entries = []
                    for row in cursor.fetchall():
                        entries.append({
                            "id": row["id"],
                            "patient_id": row["patient_id"],
                            "session_id": row["session_id"],
                            "content": row["content"],
                            "entry_type": row["entry_type"],
                            "timestamp": row["timestamp"],
                            "metadata": json.loads(row["metadata"] or "{}")
                        })
                
                return {
                    "success": True,
                    "patient_id": patient_id,
                    "entries": entries,
                    "total_count": len(entries)
                }
            
            except Exception as e:
                logger.error(f"환자 메모리 조회 오류: {e}")
                return {"success": False, "error": str(e)}
    
    async def start_server(self):
        """FastAPI 서버를 시작합니다."""
        logger.info(f"메모리 FastAPI 서버를 포트 {self.port}에서 시작합니다...")
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """메인 함수"""
    server = MemoryServer(port=8081)
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main()) 