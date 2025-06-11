import os
import signal
import subprocess
import sys
import time


def main():
    processes = {}
    
    # 스크립트가 위치한 디렉토리 (medical_agent)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # agents 디렉토리
    agents_dir = os.path.dirname(current_dir)
    
    # 실행할 스크립트의 절대 경로 설정
    scripts = {
        "mcp_servers": os.path.join(current_dir, "start_mcp_servers.py"),
        "masking_agent": os.path.join(agents_dir, "masking_agent", "start_masking_agent.py"),
        "medical_agent": os.path.join(current_dir, "__main__.py")
    }
    
    print("🚀 모든 에이전트와 서버를 시작합니다...")
    
    try:
        # MCP 서버 시작
        processes["mcp_servers"] = run_script(scripts["mcp_servers"], "mcp_servers")
        time.sleep(3)  # 서버가 시작될 시간을 줍니다.

        # Masking 에이전트 시작
        processes["masking_agent"] = run_script(scripts["masking_agent"], "masking_agent")
        time.sleep(3)

        # Medical 에이전트 (A2A 서버) 시작
        processes["medical_agent"] = run_script(scripts["medical_agent"], "medical_agent")
        time.sleep(3)
        
        print("\n✅ 모든 서비스가 시작되었습니다.")
        print("   - MCP Servers: http://localhost:8080, http://localhost:8081")
        print("   - Masking Agent: http://localhost:8000")
        print("   - Medical Agent (A2A): http://localhost:10001")
        print("\nℹ️  Ctrl+C를 눌러 모든 서비스를 종료하세요.")
        
        # 모든 프로세스가 종료될 때까지 대기
        while True:
            for name, process in processes.items():
                if process.poll() is not None:
                    print(f"--- ❌ [{name}] 프로세스가 예기치 않게 종료되었습니다. ---")
                    print(f"--- 반환 코드: {process.returncode} ---")
                    print(f"--- 위 로그에서 오류의 원인을 확인하세요. ---")
                    raise RuntimeError(f"{name} 프로세스 실행 실패")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 종료 신호를 감지했습니다. 모든 서비스를 종료합니다...")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        for name, process in processes.items():
            if process.poll() is None:
                print(f"[{name}] 종료 중...")
                process.terminate()  # SIGTERM 전송
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()  # 5초 후에도 종료되지 않으면 강제 종료
                print(f"[{name}] 종료 완료.")
        print("\n🏁 모든 서비스가 안전하게 종료되었습니다.")


def run_script(script_path, name):
    """지정된 스크립트를 서브프로세스로 실행합니다."""
    # 각 스크립트의 디렉토리를 작업 디렉토리로 설정
    cwd = os.path.dirname(script_path)
    
    # Python 실행 파일 경로
    python_executable = sys.executable

    command = [python_executable, script_path]
    if name == "medical_agent":
        command.extend(["--port", "10001"])

    print(f"[{name}] CWD: {cwd}")
    print(f"[{name}] Command: {' '.join(command)}")

    # 서브프로세스 시작
    process = subprocess.Popen(
        command,
        cwd=cwd
        # stdout, stderr를 파이프로 연결하지 않으면 부모 프로세스의 터미널에 바로 출력됩니다.
    )
    return process


if __name__ == "__main__":
    main() 