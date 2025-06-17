import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="音乐热歌进化史",
    page_icon="🎵",
    layout="wide",
)

@st.cache_data
def load_data():
    """
    Loads and preprocesses the Spotify dataset.
    This function is cached to improve performance.
    """
    df = pd.read_csv('data/data.csv')
    df['decade'] = (df['year'] // 10) * 10
    df = df[df['decade'] >= 1960]
    return df

data = load_data()


st.sidebar.header("筛选器")
selected_decades = st.sidebar.multiselect(
    '选择年代进行对比 (可多选)',
    options=sorted(data['decade'].unique()),
    default=sorted(data['decade'].unique())
)

filtered_data = data[data['decade'].isin(selected_decades)]
st.title("🎵 音乐热歌进化史")
st.markdown("""
### 从 The Beatles 到 Billie Eilish
这个交互式 Web 应用探索了从上世纪60年代至今，流行音乐的热门歌曲特征是如何演变的。
分析了 Spotify 上数十万首歌曲的**舞蹈性 (Danceability)**、**能量 (Energy)**、**正面情绪 (Valence)** 等关键音频特征。
使用侧边栏的筛选器来深入探索不同年代的音乐特征。
""")

st.header("📈 关键特征随时间演变")
st.markdown("""观察主流音乐的"性格"是如何随年份变化的。""")

yearly_avg_features = filtered_data.groupby('year')[['danceability', 'energy', 'valence', 'acousticness']].mean().reset_index()

fig_trends = px.line(
    yearly_avg_features,
    x='year',
    y=['danceability', 'energy', 'valence', 'acousticness'],
    title='音乐特征的年度平均值 (1960-2020)',
    labels={'value': '平均值', 'year': '年份', 'variable': '特征'},
    color_discrete_map={
        "danceability": "#1DB954",
        "energy": "#F9A825",
        "valence": "#2196F3",
        "acousticness": "#607D8B"
    }
)
fig_trends.update_layout(legend_title_text='音频特征')
st.plotly_chart(fig_trends, use_container_width=True)

st.markdown("""
**解读:**
- **舞蹈性 (Danceability)**: 整体呈上升趋势，现代音乐越来越适合跳舞。
- **能量 (Energy)**: 经历了波动，在80年代达到顶峰后有所回落。
- **正面情绪 (Valence)**: 似乎在缓慢下降，或许意味着现代流行乐的情感表达更加复杂或偏向忧郁。
- **原声性 (Acousticness)**: 显著下降，电子乐器和制作技术在音乐中扮演了越来越重要的角色。
""")

st.divider()
st.header("🔬 年代特征雷达图 & 热门歌曲")
st.markdown("""每个年代的音乐都有其独特的"指纹"。通过雷达图可以清晰地看到不同年代音乐风格的差异。""")

col1, col2 = st.columns([1, 2])
with col1:
    selected_decade_radar = st.selectbox(
        '选择一个年代查看其音乐"指纹"',
        options=sorted(filtered_data['decade'].unique()),
        index=len(sorted(filtered_data['decade'].unique())) - 1
    )

decade_features = filtered_data[filtered_data['decade'] == selected_decade_radar][['danceability', 'energy', 'valence', 'acousticness', 'speechiness']].mean()
fig_radar = go.Figure() 
scaler = MinMaxScaler()
radar_values = scaler.fit_transform(decade_features.values.reshape(-1, 1)).flatten()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values,
    theta=decade_features.index,
    fill='toself',
    name=f'{selected_decade_radar}年代',
    line_color='#1DB954'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=False,
    title=f'{selected_decade_radar}年代音乐特征雷达图'
)

with col1:
    st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader(f"🎵 {selected_decade_radar}年代最热门的5首歌曲")
    top_songs = filtered_data[filtered_data['decade'] == selected_decade_radar].sort_values(by='popularity', ascending=False).head(5)
    top_songs['artists'] = top_songs['artists'].apply(lambda x: x.strip("[]").replace("'", ""))
    
    for i, row in top_songs.iterrows():
        st.markdown(f"**{row['name']}** by *{row['artists']}* (流行度: {row['popularity']})")


st.divider()
st.header("💞 特征之间的关系")
st.markdown("音乐的各个维度是如何相互影响的？选择两个特征，探索它们之间的相关性。")

col_x, col_y = st.columns(2)
feature_options = ['danceability', 'energy', 'valence', 'acousticness', 'loudness', 'tempo', 'popularity']
x_axis = col_x.selectbox('选择X轴特征', feature_options, index=0)
y_axis = col_y.selectbox('选择Y轴特征', feature_options, index=1)
scatter_data = filtered_data.sample(n=2000, random_state=42) if len(filtered_data) > 2000 else filtered_data
fig_scatter = px.scatter(
    scatter_data,
    x=x_axis,
    y=y_axis,
    color='decade',
    hover_name='name',
    title=f'{y_axis.capitalize()} vs. {x_axis.capitalize()}',
    color_continuous_scale=px.colors.sequential.Viridis
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("""
**探索:**
- `energy` vs `loudness`: 高能量的歌曲通常也更响亮。
- `valence` vs `danceability`: 快乐的歌曲更适合跳舞？
- `popularity` vs `danceability`: 更具舞蹈性的歌曲更受欢迎？
""")

st.sidebar.info(
    "**项目名称**: 音乐热歌进化史\n\n"
    "**作者**: 单梓森-2212423\n\n"
    "**数据来源**: [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)"
)
