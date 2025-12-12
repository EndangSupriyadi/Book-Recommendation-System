# Laporan Proyek Rekomendasi Machine Learning - Endang Supriyadi

## Domain Proyek
Membaca merupakan aktivitas penting untuk memperluas wawasan, meningkatkan kemampuan berpikir kritis, serta memperkaya pengetahuan seseorang. Namun, di era digital saat ini, minat membaca masyarakat cenderung menurun karena banyaknya distraksi dari berbagai platform hiburan digital [1].

Di sisi lain, muncul fenomena information overload pada platform penyedia buku (perpustakaan digital, marketplace, dan aplikasi bacaan), di mana koleksi buku yang sangat besar membuat pengguna kesulitan menemukan buku yang sesuai dengan preferensinya. Tanpa sistem rekomendasi yang tepat, pengguna harus menelusuri ribuan pilihan secara manual yang tentu tidak efisien.

Karena itulah sistem rekomendasi buku menjadi hal penting untuk membantu pembaca menemukan buku yang relevan, personal, dan sesuai riwayat minat mereka. Sistem rekomendasi juga dapat membantu pembaca baru (cold-start users) mendapatkan rekomendasi meski belum memiliki riwayat bacaan.

Dengan memanfaatkan dataset Book Recommendation Dataset (Kaggle), proyek ini membangun dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering untuk memberikan rekomendasi yang lebih akurat dan sesuai kebutuhan pengguna.


## _Business Understanding_
1. Problem Statements
- Bagaimana membangun sistem rekomendasi berbasis content-based filtering yang mampu memberikan rekomendasi buku mirip berdasarkan karakteristik konten seperti penulis, judul, atau kata kunci?
- Bagaimana memberikan rekomendasi kepada pengguna baru atau pengguna yang ingin membaca buku yang berbeda menggunakan pendekatan collaborative filtering berbasis pola rating dari pengguna lain?

2. Goals
- Menghasilkan rekomendasi buku yang dipersonalisasi berdasarkan kemiripan konten melalui content-based filtering.
- Menghasilkan rekomendasi buku menggunakan collaborative filtering untuk membantu pengguna baru atau pengguna dengan sedikit riwayat bacaan.
- Meningkatkan relevansi dan kualitas rekomendasi agar pengguna dapat mencari buku lebih efisien.

3. Solution Approach
Sistem rekomendasi pada proyek ini dikembangkan melalui dua pendekatan:

A. Content-Based Filtering
- Mengekstraksi fitur dari judul dan penulis menggunakan TF-IDF.
- Mengubah teks menjadi vektor numerik.
- Mengukur kemiripan antar buku menggunakan cosine similarity.
- Memberikan rekomendasi berdasarkan buku yang sebelumnya disukai pengguna.

Kelebihan:
- Rekomendasi sangat personal.
- Tidak bergantung pada rating pengguna lain.

Kekurangan:
- Rekomendasi terbatas pada item yang mirip saja.

B. Collaborative Filtering
- Menggunakan data rating dari pengguna lain untuk mengidentifikasi pola minat.
- Melatih model embedding (user–item) dan menghitung prediksi rating menggunakan dot product.
- Menerapkan fungsi aktivasi sigmoid dan loss function Binary Crossentropy.

Kelebihan:
- Dapat merekomendasikan buku yang tidak mirip (diversity tinggi).
- Cocok untuk cold-start item.

Kekurangan:
- Membutuhkan banyak data rating.
- Tidak bekerja optimal jika banyak nilai 0 (implicit).

## _Data Understanding_

Dataset _Book Recommendation Dataset_ ini memiliki 3 file
1. Users
   Berisi para pengguna. Perhatikan bahwa ID pengguna (User-ID) telah dianonimkan dan dipetakan ke bilangan integer. Data demografis disediakan (_Location, Age_) jika tersedia. Jika tidak, bidang-bidang ini berisi nilai NULL.  <br>
_User-ID_ = nomer identitas diri dalam dataset <br>
_Location_ = Lokasi dari _users_  <br>
_Age_ = Umur dari _users_  <br>

2. Books
   Buku diidentifikasi dengan ISBN masing-masing. ISBN yang tidak valid telah dihapus dari kumpulan data. Selain itu, beberapa informasi berbasis konten diberikan (_Book-Title, Book-Author, Year-Of-Publication, Publisher_), yang diperoleh dari Amazon Web Services. Perhatikan bahwa jika ada beberapa pengarang, hanya pengarang pertama yang disediakan. URL yang menautkan ke gambar sampul juga diberikan, muncul dalam tiga rasa yang berbeda (_Image-URL-S, Image-URL-M, Image-URL-L_), yaitu kecil, sedang, besar. URL ini mengarah ke situs web Amazon.  <br>
ISBN = nomer identitas buku  <br>
_Book-Title_ = Judul Buku <br>
_Book-Author_ = Penulis Buku  <br>
_Year-of-Publication_ = Tahun Publikasi  <br>
_Publisher_ = Lembaga Publisher  <br>
_Image-URL-S, Image-URL-M, Image-URL-L_ = Image Url yang mengarah pada situs web Amazon  <br>

3. Ratings
   Berisi informasi peringkat buku. _Ratings_ (_Book-Rating_) dapat berupa nilai eksplisit, yang dinyatakan dalam skala 1-10 (nilai yang lebih tinggi menunjukkan apresiasi yang lebih tinggi), atau implisit, yang dinyatakan dengan angka 0.  <br>
_User-Id_ = identitas user yang memberikan ratings  <br>
_Book-Ratings_ = rating yang dimiliki buku  <br>

dataset buku ada 271360 _rows_ dan 8 _columns_ <br>
dataset rating ada 1149780 _rows_ dan 3 _columns_ <br>
dataset user ada 278858 _rows_ dan 3 _columns_ <br>

Link Dataset :
sumber dataset https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset




### _Eksploratory Data_
- Membaca dataset <br>
Pertama arahkan alamat path penyimpanan dataset , setalah itu membaca dan menampilkan data dengan "read_csv" pastikan dataset berformat csv. untuk melihat jumlah data dalam dataset gunakan "dataset.shape". 

1. Membaca Dataset rating <br>
   Dalam Tabel 1 merupakan 5 data teratas dari isi dataset rating yang terdiri dari kolom _User-Id_, ISBN, dan _Book-Rating_ ,dalam tabel ada yang memiliki _book-rating_ yaitu 0 karena 0 termasuk nilai maka dapat digunakan dalam pemrosesan data. Di dataset Rating terdapat _User-Id_ yang sama disini berarti satu _user_ merating beberapa buku.

   Tabel 1. Dataset Rating <br>

   | | **User-Id** | **ISBN** | **Book-Rating** |
   |-|-------------|----------|-----------------|
   |0|276725       |034545104X|0                |
   |1|276726       |0155061224|5                |
   |2|276727       |0446520802|0                |
   |3|276729       |052165615X|3                |
   |4|276729       |0521795028|6                |
  
2. Melihat Dataset book <br>
   Pada tabel 2 berisi 5 data teratas dari dataset _book_ yang terdiri dari ISBN, _Book-Title_, _Book-Author_, _Year-Of-Publication_ dan _Publisher_ , ISBN disini sebagai kunci pembeda dengan data yang lainnya maka ISBN berbeda pada setiap datanya<br>
   Tabel 2 Dataset book <br>

   
   | | **ISBN** | **Book-Title**              | **Book-Author**     | **Year-Of-Publication** | **Publisher**            |	
   |-|----------|-----------------------------|---------------------|-------------------------|--------------------------|
   |0|0195153448|Classical Mythology          |Mark P. O. Morford   |2002                     |Oxford University Press   |
   |1|0002005018|Clara Callan                 |Richard Bruce Wright |2001                     |HarperFlamingo Canada     |
   |2|0060973129|Decision in Normandy         |Carlo D'Este         |1991                     |HarperPerennial           |
   |3|0374157065|Flu: The Story of the Great..|Gina Bari Kolata     |1999                     |Farrar Straus Giroux	    |
   |4|0393045218|The Mummies of Urumchi       |E. J. W. Barber	     |1999                     |W. W. Norton &amp; Company|
<br>

pada dataset ini terdapat <br>
Total Baris Awal: 271360 <br>
Jumlah Baris Unik (berdasarkan Title & Author): 251185 <br>
Jumlah Baris Duplikat yang Akan Dihapus: 20175 <br>


   Pada gambar 1 yaitu memvisualisasikan dan meneliti distribusi rating dataframe. Rating terbanyak adalah 0 dan terendah adalaah 1, disini berarti Rating 0 menunjukkan pengguna belum memberikan rating eksplisit (implicit feedback)<br>


<img width="364" height="278" alt="image" src="https://github.com/user-attachments/assets/af07c2b4-1890-49b7-9764-a99439b29a53" />
<br />
gambar 1. Distribusi Rating <br>

Pada gambar 2 memvisualisasikan dan meneliti distribusi tahun terbitnya buku dari book dataframe disini data tahun terbit cenderung meningkat setiap tahunnya. Terbanyak tahun 2002, dataset ini terdiri dari tahun 1955 sampai dengan 2002. disini berarti semakin banyak orang membaca buku setiap tahunnya<br>


<img width="395" height="213" alt="image" src="https://github.com/user-attachments/assets/7c6633f1-b1da-4a04-a264-edaba8bc6c87" />

gambar 2. Distribusi tahun terbitnya buku <br>

#### Cek Missing Value

Gambar 3 

<img width="117" height="193" alt="image" src="https://github.com/user-attachments/assets/bae5fa64-a844-4f90-9360-2c57db9b5a3c" />

Gambar 3 Missing value dari dataset book

Gambar 4 

<img width="83" height="83" alt="image" src="https://github.com/user-attachments/assets/c7eb3bbc-cb0e-4cc8-ba45-3f245ea70028" />

Gambar 4 Missing value dari book rating


## _Data Preparation_
<br>
Melakukan transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan

### Handle _Missing Value_
Pada tahap ini data dicek menggunakan isnull().sum().
Dataset Books dan Users memiliki beberapa nilai kosong pada kolom seperti Age dan Publisher.
Strategi yang dilakukan:

Books: baris yang memiliki missing value pada kolom penting (Author, Title, Publisher) di-drop.

Users: nilai Age bernilai NULL diisi dengan median (karena tipe numerik dan distribusi tidak normal).

Gambar 5 

<img width="114" height="156" alt="image" src="https://github.com/user-attachments/assets/6b11f678-a3ab-4e34-856b-97122eab8ba0" />

Gambar 5 hasil drop missing value dari book dataset


### Handling Duplicates
Dataset Books memiliki:

Total baris awal: 271.360

Baris unik berdasarkan kombinasi Book-Title dan Book-Author: 251.185

Baris duplikat yang dihapus: 20.175

Duplikasi sering terjadi karena ISBN berbeda namun judul & penulis sama.

### Normalisasi nama kolom
Hal ini untuk memudahkan proses pemrosesan kolom yang diubah "User-ID" dan "Book-Rating"
dengan hasil <br>
Gambar 6

<img width="126" height="98" alt="image" src="https://github.com/user-attachments/assets/79ddb5a7-55db-4887-b202-289adc623202" />

pada gambar 6 terlihat bahwa "User-ID" dan "Book-Rating" berubah menjadi "user_id"  dan "rating"

### Encoding Label (Collaborative Filtering)

CF memerlukan label berbentuk ID numerik yang rapat (0 … n).

Oleh karena itu kolom:

User-ID → user_encoded_to_user

ISBN → book_encoded_to_book

dilakukan encoding menggunakan LabelEncoder

### TF-IDF Transformation (Content-Based)
TF-IDF yang merupakan kepanjangan dari Term Frequency-Inverse Document Frequency memiliki fungsi untuk mengukur seberapa pentingnya suatu kata terhadap kata - kata lain dalam dokumen. Umumnya menghitung skor untuk setiap kata untuk menandakan pentingnya dalam dokumen dan corpus. Metode sering digunakan dalam Information Retrieval dan Text Mining.
lalu setelah itu akan melakukan fit dan transformasi ke dalam matriks, matriks tersebut adalah tfidf_matrix. Pada Gambar 7 tfidf_matrix terdapat 10000 ukuran data dan 5575 nama penulis buku. 
<br>

<img width="71" height="14" alt="image" src="https://github.com/user-attachments/assets/33b3473b-c8db-4c15-95fe-3f0b8e1d0d91" />

Gambar 7 matriks <br>
Cara kerja TF-IDF <br>
- Men-transform teks “Book-Author” dan “Book-Title” menjadi fitur numerik.
- Menghasilkan matriks TF-IDF berukuran n_books × n_features. <br>


### Pembuatan User-Book Matrix (Collaborative Filtering)
- Mengonversi data rating menjadi pasangan user–book untuk training model embedding.

## Modeling

### _1. Content Filtered Recommendation System_

Definisi dan Cara Kerja
1. Definisi

Content-Based Filtering adalah pendekatan sistem rekomendasi yang memberikan rekomendasi berdasarkan kemiripan konten antara item (misalnya buku, film, musik).
Dalam konteks proyek ini, sistem menganalisis informasi deskriptif buku seperti:

- Judul buku
- Nama penulis
- Kata kunci penting dalam judul
- Informasi lain yang menggambarkan isi buku

Sistem kemudian merekomendasikan buku yang memiliki karakteristik mirip dengan buku yang sebelumnya disukai atau dicari oleh pengguna.

2. Cara Kerja

Content-Based Filtering bekerja melalui beberapa tahap utama:

a. Ekstraksi Fitur dari Konten Buku

Data teks seperti judul buku dan nama penulis diubah menjadi representasi numerik menggunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency).

TF-IDF membantu model memahami:

- Kata mana yang paling penting dalam deskripsi buku

- Seberapa unik kata tersebut relatif terhadap seluruh koleksi buku

Output dari proses ini adalah matriks TF-IDF yang mewakili setiap buku dalam bentuk vektor numerik.

b. Menghitung Kemiripan Antar Buku

Kemiripan dihitung menggunakan Cosine Similarity, yaitu ukuran seberapa dekat dua vektor (buku) satu sama lain.

Cosine similarity menghasilkan nilai:

- 1 → Buku sangat mirip
- 0 → Buku tidak mirip sama sekali

Dengan ini, model dapat mengetahui daftar buku yang paling dekat profilnya dengan buku input.

c. Mengambil Top-N Rekomendasi

Ketika pengguna memilih atau memasukkan satu buku:

- Sistem mengambil vektor TF-IDF buku tersebut.

- Mengurutkan semua buku lain berdasarkan tingkat kemiripannya (cosine similarity).

- Mengembalikan top-N buku dengan nilai kemiripan tertinggi.

- Buku yang sama seperti input dikeluarkan dari rekomendasi (“self-item removal”).

3. Kelebihan Content-Based Filtering

- Rekomendasi sangat personal dan relevan dengan preferensi pengguna.

- Tidak bergantung pada rating pengguna lain (tidak terpengaruh cold-start item).

- Dapat memberikan rekomendasi meskipun hanya ada satu buku yang disukai pengguna.



Nilai K pada fungsi menandakan jumlah rekomendasi yang akan ditampilkan
Atribut argpartition berguna untuk mengambil sejumlah nilai k, dalam fungsi ini 5 tertinggi dari tingkat kesamaan yang berasal dari dataframe cosine_sim_df. Kemudian, mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini dimasukkan ke dalam variabel closest. Berikutnya, menghapus book_title yang yang dicari agar tidak muncul dalam daftar rekomendasi. 

- Mencari Rekomendasi dari buku yang sudah dibaca 
Buku yang dibaca yaitu The Diaries of Adam and Eve berikut detailnya sesuai dengan gambar 8, ini menjadi buku yang dianggap sudah dibaca oleh user. dalam Cosine Similarity nanti akan mencari book yang mirip dengan The Diaries of Adam and Eve, sehingga perlu drop book_title The Diaries of Adam and Eve agar tidak muncul dalam daftar rekomendasi yang diberikan nanti. 

Gambar 8. Buku yang sudah dibaca <br>

<img width="311" height="42" alt="image" src="https://github.com/user-attachments/assets/558cb812-6f97-446a-82d6-e7f11d552017" />


<br>
selanjutnya yaitu menampilkan buku rekomendasi. mungkin beberapa kasus ada yang menampilkan rekomendasi yang sama jadi perlu dihapus jika ada rekomendasi yang sama dengan perintah "rekomendasi.drop_duplicates()". berikut ini 5 rekomendasi buku yang ada di Gambar 9, pada tabel disini ada kesamaan nama penulis bukunya berarti sistem memberikan rekomendasi buku dengan penulis yang sama dengan buku yang telah user baca. <br>


Gambar 9 lima Rekomendasi Buku<br>

<img width="267" height="96" alt="image" src="https://github.com/user-attachments/assets/b0625d91-48d1-4efe-ae7e-c5487cd66699" />





###  _2. Collaborative Filtered Recommendation System_

   Collaborative Based Filtering adalah sistem rekomendasi berdasarkan pendapat suatu komunitas.
- Kelebihan pada Collaborative Based Filtering bila dibandingkan dengan Content Based Filtering adalah pengguna dapat mengeksplorasi item atau konten di luar preferensi pengguna. Pengguna pun juga dapat mendapat rekomendasi sesuai dengan kecenderungan publik yang dianalisa lewat penilaian pengguna - pengguna lainnya.
- Kekurangan pada Collaborative Based Filtering adalah pengguna kurang mendapatkan rekomendasi sesuai preferensi pribadi. Konten - konten yang diberikan oleh sistem rekomendasi lebih banyak berasal dari preferensi publik dan bukan preferensi pribadi.Pada Collaborative Based Filtering, menggunakan penilaian dari pengguna - pengguna untuk mendapatkan rekomendasi buku - buku.

pada _Collaborative Filtered Recommendation System_ menerapkan teknik collaborative filtering untuk membuat sistem rekomendasi. Teknik ini membutuhkan data rating dari user. menghasilkan rekomendasi sejumlah buku yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna,mengidentifikasi buku-buku yang mirip dan belum pernah baca oleh pengguna untuk direkomendasikan. Menggunakan teknik _collaborative filtering_ untuk membuat rekomendasi ini. 

Pada tahap ini, model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama, melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Selain itu, dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

#### Split Data

Dataset rating dibagi menjadi data latih dan data validasi dengan proporsi 80:20 untuk mengevaluasi performa model.

#### _Binary Crossentropy_
Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer
- Mendapatkan Rekomendasi
mendefinisikan ulang book_datase dan rating_dataset
akan mengambil user_id secara acak dari rating_dataset. Dari user_id ini perlu mengetahui buku - buku apa saja yang pernah dibaca dan yang belum pernah dibaca, sehingga hanya dapat merekomendasikan buku - buku yang belum dibaca.
hasilnya seperti gambar  10 ini <br>
<br>
Gambar 10 menampilkan rekomendasi buku buku dengan _user_ : 277195. sistem memberikan rekomendasi buku yang mungkin user juga belum membacanya dan mungkin ada kesamaan dalam suatu hal bisa dari penulis yang sama, atau mungkin rating yang bagus<br>

Gambar. 10 Rekomendasi Buku <br>

<img width="491" height="135" alt="image" src="https://github.com/user-attachments/assets/dd9950d9-e5e2-4210-8908-a4fdf9fe00a3" />



## Evaluation
### _1. Content Filtered Recommendation System_
   Dalam evaluasi yang digunakan adalah precision yaitu salah satu metrik yang digunakan untuk mengukur seberapa akurat sistem rekomendasi dalam memberikan rekomendasi yang relevan kepada pengguna. membandingkan tingkat kesamaannya
metrik evaluasi 
precision: Jumlah buku yang memiliki kemiripan dalam buku yang direkomendasikan/ Jumlah buku yang direkomendasikan
penulis buku yang sudah di baca Mark Twain seperti Gambar 8, dan jumlah buku yang direkomendasikan Gambar 9
jadi precision : 5/5 = 100% sama


###  _2. Collaborative Filtered Recommendation System_

Nenggunakan root mean squared error (RMSE) sebagai metrics evaluation. 

<img width="289" height="220" alt="image" src="https://github.com/user-attachments/assets/4dbc9bf7-d168-4c8a-b6fc-edce94fdddde" />
<br>
Gambar 11 Hasil _Training model_<br>

Perhatikanlah pada gambar 11, proses training model cukup smooth dan model konvergen pada epochs sekitar 30. Dari proses ini, memperoleh nilai error akhir sebesar sekitar 0.1999 dan error pada data validasi sebesar 0.3429. 

### Kesimpulan 
1. Pendekatan content-based filtering berhasil memberikan rekomendasi buku yang relevan berdasarkan kemiripan fitur, terutama penulis.

2. Pendekatan collaborative filtering efektif memberikan rekomendasi untuk pengguna baru dan dapat menangkap pola minat dari komunitas pembaca.

3. Kedua model saling melengkapi dan bersama-sama meningkatkan kualitas sistem rekomendasi.


Referensi Jurnal : <br>
[1]	A. Suryana, I. B. Zaki, J. Sua, G. Phua, J. Jekson, and C. Celvin, “Pentingnya Membaca Buku bagi Generasi Baru di Era Teknologi Bersama Komunitas Ayobacabatam,” Natl. Conf. Community Serv. Proj., vol. 3, pp. 715–720, 2021, [Online]. Available: https://journal.uib.ac.id/index.php/nacospro/article/view/6010






