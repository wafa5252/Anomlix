# تصميم شعار SVG احترافي يُمثل هوية النظام (Anomalix+)
custom_svg_logo = """
<svg width="45" height="45" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- خلفية متدرجة دائرية ذات طابع أمني وصحي -->
  <circle cx="50" cy="50" r="48" fill="url(#bg_grad)" stroke="#38BDF8" stroke-width="2"/>
  
  <!-- خطوط نبضات القلب والموثوقية (Pulse & Anomaly Wave) -->
  <path d="M15 50H32L40 30L55 70L65 45L72 55H85" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- علامة الزائد المؤسسية في زاوية الشعار -->
  <rect x="68" y="15" width="20" height="20" rx="4" fill="#0EA5E9"/>
  <path d="M78 20V30M73 25H83" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>

  <defs>
    <linearGradient id="bg_grad" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E3A8A"/>
    </linearGradient>
  </defs>
</svg>
"""