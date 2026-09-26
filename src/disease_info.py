"""
Info penyakit per kelas — teks referensi STATIS (tanpa LLM).
Kelas tetap 5, jadi cukup lookup dict. Deterministik & bisa disitasi.

ponytail: konten ringkas hasil rangkuman umum. Untuk penelitian, VERIFIKASI &
SITASI dari sumber agronomi resmi (mis. Balitsa/BPTP, literatur fitopatologi)
sebelum dipakai — jangan andalkan teks ini apa adanya untuk saran pestisida.
"""

DISEASE_INFO = {
    "Purple Blotch": {
        "patogen": "Alternaria porri (jamur)",
        "penyebab": "Kelembapan tinggi, daun basah berkepanjangan, drainase buruk, dan sisa tanaman terinfeksi.",
        "gejala": "Bercak oval ungu/cokelat dengan halo kuning; bercak membesar dan menyatu, ujung daun mengering.",
        "penanganan": "Buang daun/sisa tanaman terinfeksi, perbaiki drainase & jarak tanam, rotasi tanaman, "
                      "aplikasi fungisida protektan (mis. mankozeb/klorotalonil) sesuai anjuran.",
    },
    "Botrytis Leaf Blight": {
        "patogen": "Botrytis squamosa (jamur)",
        "penyebab": "Cuaca lembap/dingin, embun lama pada daun, dan pengairan berlebih dari atas (overhead).",
        "gejala": "Banyak bercak kecil putih dengan tepi cokelat; jaringan mati, ujung daun hangus (blight).",
        "penanganan": "Sanitasi lahan, hindari irigasi overhead, atur jarak tanam agar sirkulasi udara baik, "
                      "fungisida sesuai anjuran saat gejala awal.",
    },
    "Downy Mildew": {
        "patogen": "Peronospora destructor (oomycete)",
        "penyebab": "Kelembapan sangat tinggi & suhu sejuk, kabut/embun, serta benih/umbi terinfeksi.",
        "gejala": "Bercak pucat difus (bukan bintik tegas), lapisan jamur abu-abu/ungu di permukaan daun, daun menguning.",
        "penanganan": "Gunakan benih sehat, kurangi kelembapan (drainase & jarak tanam), rotasi tanaman, "
                      "fungisida sistemik (mis. berbahan metalaksil) sesuai anjuran.",
    },
    "Rust": {
        "patogen": "Puccinia allii (jamur karat)",
        "penyebab": "Kelembapan tinggi, suhu sedang, dan kelebihan pemupukan nitrogen.",
        "gejala": "Pustul oranye-kemerahan menonjol seperti serbuk karat, tersebar di daun; jaringan sekitar menguning.",
        "penanganan": "Buang daun terinfeksi, hindari kelebihan nitrogen, perbaiki sirkulasi udara, rotasi tanaman, "
                      "fungisida (mis. golongan triazol) sesuai anjuran.",
    },
    "Healthy Leaf": {
        "patogen": "—",
        "penyebab": "—",
        "gejala": "Daun hijau seragam, tanpa bercak/lesi.",
        "penanganan": "Lanjutkan pemeliharaan rutin dan pemantauan berkala untuk deteksi dini.",
    },
}
