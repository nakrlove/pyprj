# %%
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# 데이터 불러오기
file_path = 'apartment_rent_data_price.csv'
df = pd.read_csv(file_path, sep=';', encoding='utf-8')
# df


# 필요한 컬럼만 선택
df = df[['시군구', '전월세구분', '보증금(만원)', '월세금(만원)', '계약년월']]
df


# '시군구'에서 '구' 정보만 추출
df['구'] = df['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
df.dropna(subset=['구'], inplace=True)


# 보증금(만원)과 월세금(만원) 컬럼의 콤마(,) 제거 및 숫자형으로 변환
df['보증금(만원)'] = df['보증금(만원)'].str.replace(',', '', regex=True).fillna('0')
df['월세금(만원)'] = df['월세금(만원)'].str.replace(',', '', regex=True).fillna('0')

df['보증금(만원)'] = pd.to_numeric(df['보증금(만원)'])
df['월세금(만원)'] = pd.to_numeric(df['월세금(만원)'])


# 보증금과 월세금 합산하여 '총 전월세 금액' 계산 (월세는 120개월로 환산)
df['총 전월세 금액'] = df['보증금(만원)'] + df['월세금(만원)'] * 120


# 비정상적인 극단값(outlier) 제거
# 예: 20억(200000만원)을 초과하는 데이터는 제외
df = df[df['총 전월세 금액'] <= 200000]


# 금액대별 분류 (1억원 단위)
bins = list(range(0, int(df['총 전월세 금액'].max()) + 10000, 10000))
labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['총 전월세 금액'].max()) + 10000, 10000)[:-1]]
df['금액대'] = pd.cut(df['총 전월세 금액'], bins=bins, labels=labels, right=False)



# 전세와 월세 데이터 분리
df_jeonse = df[df['전월세구분'] == '전세'].copy()
df_wolse = df[df['전월세구분'] == '월세'].copy()


# 구별, 금액대별 수요량 (거래 건수) 계산
demand_jeonse = df_jeonse.groupby(['구', '금액대']).size().unstack(fill_value=0)
demand_wolse = df_wolse.groupby(['구', '금액대']).size().unstack(fill_value=0)


print("전세 수요량 (일부):\n", demand_jeonse.head())
print("\n월세 수요량 (일부):\n", demand_wolse.head())

plt.figure(figsize=(20, 10))

# 전세 히트맵
plt.subplot(1, 2, 1)
sns.heatmap(demand_jeonse, annot=True, fmt='d', cmap='YlGnBu')
plt.title('구별 전세 금액대별 수요량 히트맵')
plt.xlabel('금액대')
plt.ylabel('구')

# 월세 히트맵
plt.subplot(1, 2, 2)
sns.heatmap(demand_wolse, annot=True, fmt='d', cmap='OrRd')
plt.title('구별 월세 금액대별 수요량 히트맵')
plt.xlabel('금액대(보증금+월세 환산)')
plt.ylabel('구')

plt.tight_layout()
plt.show()

# '계약년월'을 날짜 형식으로 변환
df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')

# 금액대별 분류 (1억원 단위)
bins = list(range(0, int(df['총 전월세 금액'].max()) + 10000, 10000))
labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['총 전월세 금액'].max()) + 10000, 10000)[:-1]]
df['금액대'] = pd.cut(df['총 전월세 금액'], bins=bins, labels=labels, right=False)



# 데이터에 포함된 모든 구 리스트 확인
print("데이터에 포함된 구 리스트:", df['구'].unique())


# 구별, 금액대별 수요량 계산
demand_by_gu = df.groupby(['구', '금액대']).size().unstack(fill_value=0)


# 스케일링
scaler = StandardScaler()
scaled_demand = scaler.fit_transform(demand_by_gu)


# K-Means 모델 학습
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
demand_by_gu['cluster'] = kmeans.fit_predict(scaled_demand)

# PCA를 이용한 군집화 결과 시각화
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_demand)


plt.figure(figsize=(12, 10))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=demand_by_gu['cluster'], palette='viridis', s=200)
plt.title('구별 수요량 패턴 군집화 결과 (K-Means + PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
for i, gu in enumerate(demand_by_gu.index):
    plt.text(reduced_data[i, 0] + 0.05, reduced_data[i, 1], gu)
plt.show()


# 구별 월별 총 거래 건수 시계열 데이터 생성
time_series_data = df.groupby(['계약년월', '구']).size().reset_index(name='거래건수')


# # 각 구별로 예측 수행
forecast_results = {}
gu_list = time_series_data['구'].unique()


# 예측 결과 출력
for gu, forecast_df in forecast_results.items():
    print(f"\n--- {gu} 2025년 10~12월 예측 거래량 ---")
    print(forecast_df)


plt.figure(figsize=(30, 6 * num_rows))

for i, gu in enumerate(gu_list):
    gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
    
    # Prophet 모델 학습
    m = Prophet()
    m.fit(gu_data)
    
    # 2025년 10월, 11월, 12월 예측 (3개월)
    future = m.make_future_dataframe(periods=3, freq='M')
    forecast = m.predict(future)
    
    # 예측 결과 저장
    forecast_results[gu] = forecast[['ds', 'yhat']].tail(3)
    
    # 막대 그래프용 데이터 프레임 생성
    bar_data = gu_data.copy()
    
    # 예측 데이터 추가
    forecast_df = forecast_results[gu]
    bar_data = pd.concat([bar_data, forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
    
    # 예측 데이터 구분용 컬럼 생성
    bar_data['type'] = '실제'
    bar_data.loc[bar_data['ds'].isin(forecast_df['ds']), 'type'] = '예측'
    
    # 그래프 시각화 (막대 그래프)
    ax = plt.subplot(num_rows, num_cols, i + 1)
    sns.barplot(x=bar_data['ds'].dt.strftime('%Y-%m'), y=bar_data['y'], hue=bar_data['type'], dodge=False, ax=ax)

    ax.set_title(f'{gu} 거래량 예측', fontsize=20)
    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('거래 건수', fontsize=12)
    ax.legend(title='데이터 유형', loc='upper left', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=10)

    plt.tight_layout()
    plt.show()










# %%
