#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from medical_tools import PatientSearchTool


def debug_patient_data():
    print("=== 환자 데이터 디버깅 ===")
    tool = PatientSearchTool()
    
    # 내과 환자 확인
    if '내과' in tool.patients_data:
        patients = tool.patients_data['내과']
        print(f"내과 환자 수: {len(patients)}")
        
        # 첫 번째 환자 구조 확인
        if patients:
            first_patient = patients[0]
            print("\n=== 첫 번째 환자 데이터 구조 ===")
            print("키 목록:", list(first_patient.keys()))
            
            # diagnoses vs diagnosis 확인
            if 'diagnoses' in first_patient:
                print("diagnoses 필드:", first_patient['diagnoses'])
            if 'diagnosis' in first_patient:
                print("diagnosis 필드:", first_patient['diagnosis'])
        
        # 당뇨병 환자 수동 검색
        print("\n=== 당뇨병 관련 환자 수동 검색 ===")
        diabetes_count = 0
        for i, patient in enumerate(patients[:10]):  # 처음 10명만 확인
            name = patient.get('name', '이름없음')
            
            # diagnoses 배열에서 당뇨병 찾기
            diagnoses = patient.get('diagnoses', [])
            if isinstance(diagnoses, list):
                for diag in diagnoses:
                    if isinstance(diag, dict):
                        diag_name = diag.get('name', '')
                        if '당뇨' in diag_name or 'diabetes' in diag_name.lower():
                            print(f"✅ 당뇨병 환자 발견: {name} - {diag_name}")
                            diabetes_count += 1
                            break
            
            # diagnosis 단일 필드에서 당뇨병 찾기  
            diagnosis = patient.get('diagnosis', '')
            if isinstance(diagnosis, str) and ('당뇨' in diagnosis or 'diabetes' in diagnosis.lower()):
                print(f"✅ 당뇨병 환자 발견 (단일): {name} - {diagnosis}")
                diabetes_count += 1
        
        print(f"\n총 발견된 당뇨병 환자: {diabetes_count}명")
        
        # 검색 로직 테스트
        print("\n=== 검색 로직 테스트 ===")
        result = tool._run('당뇨병환자')
        print("당뇨병환자 검색 결과:", result[:200], "...")

if __name__ == "__main__":
    debug_patient_data() 