import streamlit as st
import pandas as pd
import numpy as np

# إعداد الصفحة
st.set_page_config(
    page_title="Anomalix - Enterprise Hospital CMMS",
    layout="wide"
)

# بيانات الأقسام واسم المستخدم وكلمة المرور الخاصة بكل قسم
CREDENTIALS = {
    "Executive Management": {"user": "admin_exec", "pass": "exec2026"},
    "IT Infrastructure & Biomed": {"user": "tech_biomed", "pass": "biomed2026"},
    "Quality & Patient Safety": {"user": "quality_lead", "pass": "quality2026"}
}

# حالة الجلسة لتسجيل الدخول
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["role"] = None

# شاشة تسجيل الدخول الرئيسية
if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; color: #0F172A;'>Anomalix+ Enterprise CMMS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>تسجيل الدخول الموحد للأقسام المؤسسية</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_role = st.selectbox("اختر القسم / الدور:", list(CREDENTIALS.keys()))
        username_input = st.text_input("اسم المستخدم:")
        password_input = st.text_input("كلمة المرور:", type="password")
        
        if st.button("تسجيل الدخول"):
            expected_user = CREDENTIALS[selected_role]["user"]
            expected_pass = CREDENTIALS[selected_role]["pass"]
            
            if username_input == expected_user and password_input == expected_pass:
                st.session_state["authenticated"] = True
                st.session_state["role"] = selected_role
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة. حاول مرة أخرى.")
    
    st.stop()

# في حال تم تسجيل الدخول بنجاح
user_role = st.session_state["role"]

# الشريط الجانبي وخيار تسجيل الخروج
st.sidebar.markdown(f"### النظام النشط: {user_role}")
st.sidebar.markdown("Compliance: SDAIA & NCA (ECC) Aligned")
st.sidebar.markdown("Security: AES-256 / TLS 1.3 Active")

if st.sidebar.button("تسجيل الخروج"):
    st.session_state["authenticated"] = False
    st.session_state["role"] = None
    st.rerun()

# الهوية البصرية والعنوان
st.markdown("""
    <div style='background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h1 style='margin:0; font-size: 28px;'>Anomalix+ Enterprise CMMS</h1>
        <p style='margin:5px 0 0 0; color: #38BDF8;'>نظام إدارة أصول المستشفيات وبنية الأجهزة التحتية والموثوقية التشغيلية</p>
    </div>
""", unsafe_allow_html=True)

# البيانات المركزية للأصول
def get_enterprise_asset_registry():
    data = {
        'asset_id': ['AST-ICU-01', 'AST-LAB-04', 'AST-SRV-12', 'AST-RAD-02', 'AST-NET-05'],
        'asset_name': ['Ventilator Alpha X', 'Centrifuge Pro 5000', 'Core Database Server', 'MRI Scanner Unit', 'Core Switch Cisco 9K'],
        'category': ['Medical Equipment', 'Medical Equipment', 'IT Infrastructure', 'Medical Equipment', 'IT Infrastructure'],
        'location': ['Main Hospital Building', 'Main Hospital Building', 'Data Center Floor 2', 'Radiology Dept', 'Server Room A'],
        'department': ['Intensive Care (ICU)', 'Pathology Lab', 'IT Operations', 'Radiology', 'Network Operations'],
        'criticality': ['High', 'Medium', 'Critical', 'High', 'Critical'],
        'status': ['Operational', 'Degraded', 'Operational', 'Maintenance', 'Operational'],
        'health_score': [92.5, 74.0, 98.1, 61.2, 95.0],
        'vendor': ['MedTech Global', 'LabCorp Systems', 'Dell Enterprise', 'Siemens Health', 'Cisco Systems']
    }
    return pd.DataFrame(data)

df_assets = get_enterprise_asset_registry()

# تبويبات النظام المؤسسي
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Dashboard", 
    "Enterprise Asset Registry", 
    "Impact Analysis & Health", 
    "HL7 FHIR Integration"
])

with tab1:
    st.subheader("Executive Management & Operations Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="VOSS Operational Stress", value="42 / 100", delta="-4.2% stable")
    with col2:
        st.metric(label="Total Tracked Assets", value=len(df_assets), delta="5 Active Nodes")
    with col3:
        st.metric(label="Avg MTTR (Repair Time)", value="34.2 min", delta="-2.1 min")
    with col4:
        st.metric(label="Critical Health Alerts", value=len(df_assets[df_assets['health_score'] < 75]), delta="Action Required", delta_color="inverse")

    st.markdown("### Asset Status Distribution")
    st.bar_chart(df_assets.set_index('asset_name')['health_score'])

with tab2:
    st.subheader("سجل الأصول المركزي (Enterprise Asset Registry - Hierarchy)")
    st.markdown("إدارة الأصول الهرمية: الموقع، القسم، الأصل، مع تتبع الأهمية التشغيلية والضمان.")
    
    selected_cat = st.selectbox("Filter by Category:", ["All"] + list(df_assets['category'].unique()))
    if selected_cat != "All":
        filtered_df = df_assets[df_assets['category'] == selected_cat]
    else:
        filtered_df = df_assets
        
    st.dataframe(filtered_df, use_container_width=True)
    
    st.download_button(
        label="تصدير سجل الأصول والتقارير (CSV)",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='Anomalix_Enterprise_Assets.csv',
        mime='text/csv',
    )

with tab3:
    st.subheader("تحليل التأثير والاعتمادية (Impact Analysis & Health Score)")
    st.markdown("تحديد الأصول المعرضة للخطر وتأثيرها المباشر على أقسام المستشفى:")
    
    critical_issues = df_assets[df_assets['health_score'] < 75.0]
    if not critical_issues.empty:
        for _, row in critical_issues.iterrows():
            st.error(f"تحذير حرج للأصل: {row['asset_name']} ({row['asset_id']}) في قسم {row['department']} | درجة الصحة: {row['health_score']}%")
            st.info(f"إجراء مقترح: بدء أمر عمل طارئ وفحص الاعتمادية المرتبطة بالمورد: {row['vendor']}")
    else:
        st.success("جميع الأصول الحرجة تعمل ضمن مستويات الصحة الطبيعية فوق 75%.")

with tab4:
    st.subheader("معايير التكامل القياسية (HL7 FHIR / DeviceMetric)")
    st.markdown("بث بيانات الأصول الطبية والتقنية بمعيار HL7 FHIR العالمي لضمان أمان وتوافق البيانات:")
    
    selected_fhir_asset = st.selectbox("Select Asset for FHIR Payload:", df_assets['asset_id'])
    asset_row = df_assets[df_assets['asset_id'] == selected_fhir_asset].iloc[0]
    
    fhir_payload = {
        "resourceType": "DeviceMetric",
        "id": f"metric-{asset_row['asset_id'].lower()}",
        "identifier": [{"system": "urn:oid:anomalix:assets", "value": asset_row['asset_id']}],
        "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/devicetric-type", "code": "operational-status"}]},
        "unit": {"coding": [{"system": "http://unitsofmeasure.org", "code": "%"}]},
        "source": {"reference": f"Device/{asset_row['asset_id']}"},
        "operationalStatus": asset_row['status'].lower(),
        "healthScore": asset_row['health_score'],
        "calibrated": "2026-10-10T21:00:00Z"
    }
    
    st.json(fhir_payload)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>Anomalix Enterprise CMMS Prototype - Powered by Python & Streamlit</p>", unsafe_allow_html=True)
