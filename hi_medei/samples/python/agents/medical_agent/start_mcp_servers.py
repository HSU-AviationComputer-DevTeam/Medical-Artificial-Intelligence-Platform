#!/usr/bin/env python3
"""
MCP 서버 시작 스크립트
hi_medei 의료 AI 에이전트를 위한 MCP 서버들을 시작합니다.
"""

import asyncio
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from mcp_servers.startup import MCPServerManager


async def main():
    """MCP 서버들을 시작하는 메인 함수"""
    print("🏥 hi_medei MCP 서버들을 시작합니다...")
    print("="*60)
    
    manager = MCPServerManager()
    
    try:
        await manager.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        print("🏁 MCP 서버들이 종료되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 