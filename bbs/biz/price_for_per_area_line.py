import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from matplotlib import font_manager, rc

from bbs.utils.common_files import FileUtils
from bbs.utils.fonts import setup_matplotlib_fonts

def engine1():
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################

    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
        # exit()

    df.info()
    # 2. 데이터 전처리
    # '계약년월'과 '계약일' 컬럼을 합쳐 '거래일' 컬럼 생성
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

    # '전용면적(㎡)'을 특정 구간으로 분류 (예: 10㎡ 단위)
    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    # '전월세구분' 컬럼을 '전세'와 '월세'로 구분
    df_jeonse = df[df['전월세구분'] == '전세'].copy()
    df_wolse = df[df['전월세구분'] == '월세'].copy()

    print("데이터 전처리 완료. 각 데이터프레임의 상위 5개 행:")
    print("전세 데이터:")
    print(df_jeonse.head())
    print("\n월세 데이터:")
    print(df_wolse.head())



    ##################################################
    # 2. 전용면적에 따른 수요량 분석 및 시각화
    ##################################################
    # 3. 전용면적별 전/월세 수요량 분석
    # 면적 구간별 거래 건수 계산
    demand_jeonse = df_jeonse.groupby('면적_구간').size().reset_index(name='전세_수요량')
    demand_wolse = df_wolse.groupby('면적_구간').size().reset_index(name='월세_수요량')

    # plot_paths = [] 
    dict = {}
    dict['면적_구간_jeonse'] = demand_jeonse['면적_구간']
    dict['수요량_wolse']  = demand_wolse['월세_수요량']
    # 운영체제에 맞는 한글 폰트 설정
    # Windows
    # plt.rcParams['font.family'] = 'Malgun Gothic'
    # Mac OS
    #plt.rcParams['font.family'] = 'AppleGothic'

    # 마이너스 부호 깨짐 방지
    # plt.rcParams['axes.unicode_minus'] = False
    setup_matplotlib_fonts()
    # 폰트 설정 확인 (선택 사항)
    # print(plt.rcParams['font.family'])

    # 4. 시각화
    plt.figure(figsize=(15, 6))


    # 전세 수요량 막대 그래프
    plt.subplot(1, 2, 1)
    plt.bar(demand_jeonse['면적_구간'], demand_jeonse['전세_수요량'], color='skyblue')
    plt.title('전용면적에 따른 전세 수요량')
    plt.xlabel('전용면적 (㎡)')
    plt.ylabel('거래량')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # filename = FileUtils().FilePathName("price_for_per_area_jeonse.png")
    # plt.savefig(filename)
    # plt.close()
    # plot_paths.append(f'/static/images/{filename}')
    

    dict['면적_구간_wolse'] = demand_wolse['면적_구간']
    dict['월세_wolse']  = demand_wolse['월세_수요량']
    # 월세 수요량 막대 그래프
    plt.subplot(1, 2, 2)
    plt.bar(demand_wolse['면적_구간'], demand_wolse['월세_수요량'], color='salmon')
    plt.title('전용면적에 따른 월세 수요량')
    plt.xlabel('전용면적 (㎡)')
    plt.ylabel('거래량')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    # filename = FileUtils().FilePathName("price_for_per_area_wolse.png")
    # plt.savefig(filename)
    # plt.close()'
    
    #차트이미지 생성하는 메소드호출 함 
    filename = FileUtils().savePngToPath("price_for_per_area_jeonse_wolse.png",closeFlag=True)
    print(f"=== 생성된 이미지 파일 위치 {filename}")
    # plot_paths.append(filename)
    dict['img_wolse']  = filename

    ##################################################
    # 3. 시계열 데이터 가공 및 10월 수요량 예측
    ##################################################

    # 5. 시계열 데이터 생성
    # 월별 거래량 집계
    monthly_demand_jeonse = df_jeonse.groupby(df_jeonse['거래일'].dt.to_period('M')).size().reset_index(name='전세_수요량')
    monthly_wolse_demand = df_wolse.groupby(df_wolse['거래일'].dt.to_period('M')).size().reset_index(name='월세_수요량')

    monthly_demand_jeonse['거래일'] = monthly_demand_jeonse['거래일'].dt.to_timestamp()
    monthly_wolse_demand['거래일'] = monthly_wolse_demand['거래일'].dt.to_timestamp()

    dict['거래일_jeonse'] = monthly_demand_jeonse['거래일']
    dict['거래일_wolse'] = monthly_wolse_demand['거래일']

    # 6. 예측 모델 학습 (선형 회귀)
    # 데이터를 숫자형으로 변환 (월 순서)
    monthly_demand_jeonse['month_index'] = np.arange(len(monthly_demand_jeonse))
    monthly_wolse_demand['month_index'] = np.arange(len(monthly_wolse_demand))

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date = pd.to_datetime('2025-10-01')
    last_date = monthly_demand_jeonse['거래일'].max()
    predict_month_index = len(monthly_demand_jeonse) + (predict_date.year - last_date.year) * 12 + (predict_date.month - last_date.month)
    print(f" ################################################# ")
    print(f" predict_month_index {predict_month_index}")
    print(f" ################################################# ")
    # 전세 수요량 예측
    X_jeonse = monthly_demand_jeonse[['month_index']]
    y_jeonse = monthly_demand_jeonse['전세_수요량']
    model_jeonse = LinearRegression()
    model_jeonse.fit(X_jeonse, y_jeonse)
    predicted_jeonse_demand = model_jeonse.predict([[predict_month_index]])

    # 월세 수요량 예측
    X_wolse = monthly_wolse_demand[['month_index']]
    y_wolse = monthly_wolse_demand['월세_수요량']
    model_wolse = LinearRegression()
    model_wolse.fit(X_wolse, y_wolse)
    predicted_wolse_demand = model_wolse.predict([[predict_month_index]])

    # 7. 결과 출력
    print(f"\n2025년 10월 전세 수요량 예측: {int(predicted_jeonse_demand[0])} 건")
    print(f"2025년 10월 월세 수요량 예측: {int(predicted_wolse_demand[0])} 건")

    # 예측 결과를 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(monthly_demand_jeonse['거래일'], monthly_demand_jeonse['전세_수요량'], marker='o', label='실제 전세 수요량')
    plt.plot(predict_date, predicted_jeonse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
    plt.title('월별 전세 수요량 및 예측')
    plt.xlabel('날짜')
    plt.ylabel('거래량')
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(monthly_wolse_demand['거래일'], monthly_wolse_demand['월세_수요량'], marker='o', label='실제 월세 수요량')
    plt.plot(predict_date, predicted_wolse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
    plt.title('월별 월세 수요량 및 예측')
    plt.xlabel('날짜')
    plt.ylabel('거래량')
    plt.legend()
    plt.grid(True)
    
    # 이미지 파일로 저장
    print("6. 구별, 금액대별 수요량 분석 및 시각화 성공.")
    # plt.tight_layout()
    # plt.show()
    filename = FileUtils().savePngToPath("price_for_per_area_jeonse_wolse-1.png",closeFlag=True)
    # plot_paths.append(filename)
    dict['img_jeonse']  = filename
    return dict


# LinearRegression(선형 회귀)
def LRegression():

    setup_matplotlib_fonts()
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################
    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
    # 데이터 전처리
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index'를 전체 데이터 기준으로 일관되게 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

    print("--- 면적 구간별 LinearRegression 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # LinearRegression 모델 학습
            model = LinearRegression()
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    # plt.show()

    return predicted_results ,performance_metrics


# Decision Tree Regression (결정 트리 회귀): 
# 데이터를 특정 조건에 따라 나누는 '결정 트리' 구조를 사용하여 예측합니다.
def DTreeRegressor():

    # 운영체제에 맞는 한글 폰트 설정
    # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
    # Mac OS
    plt.rcParams['font.family'] = 'AppleGothic'

    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
     # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
        # exit()
    # 데이터 전처리
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index'를 전체 데이터 기준으로 일관되게 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

    print("--- 면적 구간별 Decision Tree Regression 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # DecisionTreeRegressor 모델 학습
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 특성 중요도 (여기서는 'month_index' 하나이므로 1.0)
            feature_importance = model.feature_importances_[0]

            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
            analysis_metrics[rent_type][area_label] = {'특성 중요도': feature_importance}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    # plt.show()
    return predicted_results ,performance_metrics

#  Tree-based Models (트리 기반 모델)


from sklearn.tree import DecisionTreeRegressor
def engine(actionType):
    print(f" ACTION TYPE ====== {actionType}")
    if actionType == "1" :
        # LinearRegression(선형 회귀)
        return LRegression()
    elif actionType == "2" :  
        return DTreeRegressor()
    elif actionType == "3" :
        return {}   
    else :
        return {}

