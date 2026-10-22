import joblib
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title='TalentPulse - Enterprise HR Intelligence',
    page_icon='👔',
    layout='centered',
)

# --- Ultra-Modern Custom CSS (Gradient Background & Glass Cards) ---
st.markdown(
    """
    <style>
    /* Dark Gradient Modern Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }

    /* Main Container Glassmorphism Effect */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        padding: 10px !important;
    }

    /* Modern Glowing Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 17px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 15px rgba(142, 84, 233, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(142, 84, 233, 0.6) !important;
    }

    /* Subheaders and Headings */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Custom Metric Styling */
    div[data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-weight: bold !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.title('👔 TalentPulse: Enterprise HR Intelligence')
st.caption(
    '🚀 Advanced AI-Driven Employee Career Analytics & Workforce Decision'
    ' Platform'
)
st.markdown('---')


# --- Load Model & Scaler ---
@st.cache_resource
def load_artifacts():
  model = joblib.load('talentpulse_model.pkl')
  scaler = joblib.load('talentpulse_scaler.pkl')
  return model, scaler


model, scaler = load_artifacts()

# --- Input Card ---
with st.container(border=True):
  st.subheader('📝 Employee Metrics & Evaluations')

  col1, col2 = st.columns(2)

  with col1:
    perf_rating = st.slider(
        '⭐ Performance Rating (1-5)', min_value=1, max_value=5, value=4
    )
    projects_completed = st.number_input(
        '📁 Projects Completed', min_value=0, max_value=50, value=17
    )
    training_hours = st.number_input(
        '🎯 Training Hours', min_value=0, max_value=100, value=25
    )
    tenure = st.slider(
        '⏳ Tenure (Years in Company)', min_value=1, max_value=20, value=5
    )

  with col2:
    monthly_hike = st.number_input(
        '📈 Last Hike Percentage (%)', min_value=0.0, max_value=40.0, value=10.0
    )
    attendance_rate = st.slider(
        '📅 Attendance Rate (%)', min_value=0.0, max_value=100.0, value=95.0
    )
    manager_feedback = st.slider(
        '💬 Manager Feedback Score (1-5)',
        min_value=1,
        max_value=5,
        value=5,
    )

  st.markdown('')
  predict_clicked = st.button(
      '⚡ Analyze Career Decision', use_container_width=True
  )

# --- Output Result & Metrics ---
if predict_clicked:
  input_df = pd.DataFrame(
      [
          {
              'PerformanceRating': perf_rating,
              'ProjectsCompleted': projects_completed,
              'TrainingHours': training_hours,
              'Tenure': tenure,
              'MonthlyHike': monthly_hike,
              'AttendanceRate': attendance_rate,
              'ManagerFeedback': manager_feedback,
          }
      ]
  )

  input_scaled = scaler.transform(input_df)
  pred = model.predict(input_scaled)[0]
  pred_proba = model.predict_proba(input_scaled)[0]

  st.markdown('###')

  with st.container(border=True):
    st.subheader('📊 AI Recommendation Summary')

    # Quick Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric('Overall Rating', f'{perf_rating}/5')
    m2.metric('Attendance', f'{attendance_rate}%')
    m3.metric('Projects Delivered', projects_completed)

    st.markdown('---')

    if pred == 2:
      st.success(
          '🌟 **RECOMMENDATION: PROMOTION**\n\n'
          'Employee demonstrates high leadership capability and consistent'
          f' performance. (Model Confidence: **{pred_proba[2] * 100:.2f}%**)'
      )
    elif pred == 1:
      st.info(
          '⚖️ **RECOMMENDATION: STABLE / RETAIN**\n\n'
          'Employee performs reliably within current expectations. Maintain'
          f' standard trajectory. (Model Confidence: **{pred_proba[1] * 100:.2f}%**)'
      )
    else:
      st.error(
          '⚠️ **RECOMMENDATION: PIP / DEMOTION WARNING**\n\n'
          'Performance metrics are below standard thresholds. Immediate review or'
          ' Performance Improvement Plan required. (Risk Confidence:'
          f' **{pred_proba[0] * 100:.2f}%**)'
      )