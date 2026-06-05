import streamlit as st
from pathlib import Path
from PIL import Image

# ─── Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DermSight EDA",
    page_icon="🔬",
    layout="wide",
)

ASSETS = Path("assets")

# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🔬 DermSight")
st.sidebar.caption("Skin Disease Classification — EDA Dashboard")
page = st.sidebar.radio("Navigasi", [
    "Overview",
    "Class Distribution",
    "Imbalance Ratio",
    "Image Dimensions",
    "Color Mode",
    "RGB Analysis",
    "Split Ratio",
    "Summary & Conclusion",
])
st.sidebar.divider()
st.sidebar.info("Dataset: SkinDisease\n22 kelas · 15.444 gambar")


def show_chart(filename, caption=None):
    path = ASSETS / filename
    if path.exists():
        st.image(Image.open(path), use_column_width=True, caption=caption)
    else:
        st.warning(f"Chart tidak ditemukan: {filename}")


def insight_box(text):
    st.info(text)


# ─── Pages ────────────────────────────────────────────────────
if page == "Overview":
    st.title("🔬 DermSight — EDA Dashboard")
    st.markdown("Exploratory Data Analysis untuk proyek deteksi penyakit kulit berbasis Computer Vision.")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gambar",  "15.444")
    col2.metric("Total Kelas",   "22")
    col3.metric("Train Images",  "13.898")
    col4.metric("Test Images",   "1.546")

    st.divider()
    st.subheader("Latar Belakang")
    st.markdown("""
    Penyakit kulit menular ringan seperti impetigo, scabies, kurap, panu, dan eksim masih menjadi
    masalah kesehatan umum, terutama di daerah **3T (Tertinggal, Terdepan, Terluar)**. Proyek
    **DermSight** hadir sebagai solusi deteksi dini berbasis AI yang dapat diakses masyarakat
    secara mandiri, sejalan dengan **SDGs Goal 3.3**.
    """)

    st.subheader("Business Questions")
    st.markdown("""
    1. Sejauh mana model CNN dapat mengklasifikasikan penyakit kulit dari gambar secara akurat?
    2. Seberapa relevan rekomendasi OBT berdasarkan hasil klasifikasi?
    3. Apakah solusi dapat diakses masyarakat daerah 3T dengan perangkat terbatas?
    """)


elif page == "Class Distribution":
    st.title("📊 Class Distribution")
    st.caption("Distribusi jumlah gambar per kelas pada split Train dan Test.")
    show_chart("eda_01_class_distribution.png")
    insight_box(
        "Distribusi kelas pada train dan test konsisten — menandakan pembagian split dilakukan "
        "secara proporsional (stratified). Unknown_Normal menjadi kelas dominan (1.651 train), "
        "sementara Candidiasis paling minim (248 train). Lima kelas di bawah 400 gambar "
        "memerlukan augmentasi agar model tidak underfit."
    )


elif page == "Imbalance Ratio":
    st.title("⚖️ Class Imbalance Ratio")
    st.caption("Rasio ketimpangan jumlah data antar kelas relatif terhadap kelas terbesar.")
    show_chart("eda_02_imbalance_ratio.png")
    insight_box(
        "4 kelas masuk kategori severe imbalance (>5×): Candidiasis, Rosacea, Lupus, dan "
        "Sun_Sunlight_Damage. 15 kelas mild (2–5×), dan hanya 3 kelas balanced (≤2×). "
        "Class weighting atau oversampling wajib diterapkan saat training."
    )


elif page == "Image Dimensions":
    st.title("🖼️ Image Dimension Analysis")
    st.caption("Analisis dimensi dan aspect ratio gambar pada Train split.")
    show_chart("eda_03_dimensions.png")
    insight_box(
        "Dimensi gambar sangat bervariasi, dari ratusan hingga >2000px. Aspect ratio dominan "
        "adalah 1.5× (landscape), bukan square. Resize ke 224×224 dengan padding (letterbox) "
        "disarankan untuk menghindari distorsi geometri pada gambar."
    )


elif page == "Color Mode":
    st.title("🎨 Color Mode Distribution")
    st.caption("Distribusi mode warna gambar pada Train split.")
    show_chart("eda_04_color_mode.png")
    insight_box(
        "Seluruh 1.100 sampel gambar training memiliki color mode RGB. Tidak diperlukan "
        "konversi mode warna dalam pipeline preprocessing, sehingga mengurangi beban "
        "komputasi saat persiapan data."
    )


elif page == "RGB Analysis":
    st.title("🌈 Mean RGB Channel Intensity")
    st.caption("Rata-rata intensitas channel R, G, B per kelas pada Train split.")
    show_chart("eda_06_rgb_channels.png")
    insight_box(
        "Seluruh kelas didominasi channel Red — wajar untuk gambar kulit manusia. "
        "Psoriasis memiliki intensitas paling rendah (~83B), mengindikasikan gambar lebih gelap. "
        "Profil RGB antar kelas relatif mirip, sehingga model harus mengekstrak fitur "
        "tekstur dan bentuk lesi, bukan sekadar warna."
    )


elif page == "Split Ratio":
    st.title("📐 Train / Test Split Ratio")
    st.caption("Proporsi train dan test per kelas dalam persentase.")
    show_chart("eda_07_split_ratio.png")
    insight_box(
        "Rasio train/test seluruh kelas sangat konsisten di sekitar 90:10. "
        "Pembagian split dilakukan secara stratified dan merata, memastikan evaluasi "
        "model pada test set representatif untuk semua kelas."
    )


elif page == "Summary & Conclusion":
    st.title("📋 Summary & Conclusion")
    st.divider()

    st.subheader("BQ 1 — Akurasi Model CNN")
    st.markdown("""
    Dataset memiliki class imbalance signifikan (4 kelas severe, 15 mild) dan dimensi gambar
    yang tidak seragam. Untuk mencapai akurasi ≥80% dan F1-Score ≥0.75 diperlukan:
    - Resize dengan padding (letterbox) ke 224×224
    - Augmentasi data pada kelas minoritas
    - Class weighting saat training
    - Pretrained model (EfficientNet/ResNet) untuk ekstraksi fitur tekstur
    """)

    st.subheader("BQ 2 — Relevansi Rekomendasi OBT")
    st.markdown("""
    Seluruh 22 kelas terwakili dengan split proporsional — sistem rekomendasi OBT dapat
    dikembangkan untuk semua kelas. Namun 4 kelas dengan data sangat sedikit perlu disertai
    disclaimer untuk tetap berkonsultasi dengan tenaga medis.
    """)

    st.subheader("BQ 3 — Keterjangkauan Solusi")
    st.markdown("""
    Seluruh gambar sudah RGB sehingga tidak ada overhead konversi. Dengan resize konsisten
    ke 224×224, ukuran input model seragam dan ringan untuk inferensi di perangkat mobile.
    """)

    st.divider()
    st.success("Data siap diproses. Lanjut ke tahap Preprocessing & Modelling.")