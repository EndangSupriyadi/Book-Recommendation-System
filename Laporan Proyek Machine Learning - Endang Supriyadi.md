# Laporan Proyek Rekomendasi Machine Learning - Endang Supriyadi

## Domain Proyek
Membaca merupakan proses menerima dan memahami informasi yang disampaikan penulis melalui media tulisan. Aktivitas membaca memiliki peran penting dalam memperluas wawasan, meningkatkan kemampuan berpikir kritis, serta memperkuat daya ingat seseorang [1]. Namun, seiring berkembangnya teknologi dan kemudahan akses hiburan digital, minat membaca buku cenderung mengalami penurunan.

Di sisi lain, perpustakaan maupun platform penyedia buku memiliki koleksi yang sangat besar, sehingga pengguna sering mengalami kesulitan dalam menentukan buku mana yang ingin dibaca. Jumlah pilihan yang terlalu banyak (information overload) membuat pengalaman pencarian buku menjadi kurang efisien. Oleh karena itu, diperlukan sebuah sistem rekomendasi buku yang mampu membantu pengguna menemukan buku yang sesuai minat, riwayat bacaan, ataupun tren bacaan saat ini.

Sistem rekomendasi tersebut dapat meningkatkan pengalaman pengguna, mempermudah proses pencarian buku, serta membantu memperkenalkan koleksi buku yang relevan bagi pembaca baru maupun pembaca yang ingin mengeksplorasi topik berbeda.


## _Business Understanding_
1. Problem Statements

- Bagaimana membangun sistem rekomendasi yang dipersonalisasi menggunakan teknik content-based filtering agar dapat membantu pengguna menemukan buku yang memiliki karakteristik atau tipe serupa dengan riwayat bacaan sebelumnya?
- Bagaimana memberikan rekomendasi kepada pengguna yang belum memiliki riwayat bacaan (cold-start user) atau pengguna yang ingin membaca buku dengan topik berbeda menggunakan pendekatan collaborative filtering?

2. Goals

- Menghasilkan rekomendasi buku yang dipersonalisasi menggunakan content-based filtering sehingga pengguna dapat menemukan buku yang mirip dengan buku yang pernah dibaca, misalnya berdasarkan penulis, genre, atau kata kunci tertentu.
- Menghasilkan rekomendasi buku berbasis collaborative filtering bagi pengguna baru atau pengguna dengan preferensi bacaan berbeda, sehingga mereka mendapatkan rekomendasi dari pola perilaku pengguna lain yang memiliki kesamaan minat.


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


   Pada gambar 1 yaitu memvisualisasikan dan meneliti distribusi rating dataframe. Rating terbanyak adalah 0 dan terendah adalaah 1, disini berarti kebanyakan orang tidak suka bukunya sehingga memberikan rating 0<br>


<a href="https://imgbb.com/"><img src="https://i.ibb.co/6mLd8JG/download-16.png" alt="download-16" border="0"></a>
<br />
gambar 1. Distribusi Rating <br>

Pada gambar 2 memvisualisasikan dan meneliti distribusi tahun terbitnya buku dari book dataframe disini data tahun terbit cenderung meningkat setiap tahunnya. Terbanyak tahun 2002, dataset ini terdiri dari tahun 1955 sampai dengan 2002. disini berarti semakin banyak orang membaca buku setiap tahunnya<br>


<a href="https://ibb.co/0jDwJMV"><img src="https://i.ibb.co/gMjngTF/download-21.png" alt="download-21" border="0"></a><br/>
gambar 2. Distribusi tahun terbitnya buku <br>

## _Data Preparation_
<br>
Melakukan transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan

1. Cek jumlah pengguna dan buku <br>
Pada Tabel 3 terdapat jumlah pengguna: 679, jumlah buku :4688 , Min Rating : 0.0 dan Max Rating : 10.0,
didapatkan bahwa jumlah users lebih sedikit dari pada jumlah buku disini berarti kemungkinan besar orang merating lebih dari 1 buku.

<br>
Tabel 3. Menghitung jumlah User, book, min rating dan max rating <br>


|**Number of User**|**Number of Book**|**Min Rating**|**Max Rating**|
|------------------|------------------|--------------|--------------|
|679               |4688              |0.0           |10.0          |

<br>

2.  Cek Nilai _Missing Value_
Dengan" .sum() akan menampilkan data _missing value_. _missing value_ merupakan nilai yang tidak ada atau NaNN yang ada di dataset. _missing value_ bisa mempengaruhi kualiatas prediksi model sehingga harus dihapus jika data itu kecil atau mungkin tidak akan berdampak terhadap hasil model.
3. Membuang Data _Missing Value_ dan Duplikasi
Dengan menggunakan perintah "dataset.dropna()" maka data NaaN akan terhapus. lalu untuk Menghapus Duplikasi yaitu dengan menggunakan perintah "dataset.drop_duplicates()" yang nantinya jika ada row yang duplikat akan terhapus.
4. _Train Test Split_
yaitu membagi data latih dan data uji sebesar 80:20 karena perbandingan itu sangat sering digunakan dan cenderung efesien.
yang ouputnya seperti gambar 4 ini <br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/CHS1mvB/Screenshot-2024-04-01-182102.png" alt="Screenshot-2024-04-01-182102" border="0"></a>
<br>
Gambar 4 Hasil _Split Data_ <br>

## _Modeling_
### _1. Content Filtered Recommendation System_

#### TF-IDF
TF-IDF yang merupakan kepanjangan dari Term Frequency-Inverse Document Frequency memiliki fungsi untuk mengukur seberapa pentingnya suatu kata terhadap kata - kata lain dalam dokumen. Kita umumnya menghitung skor untuk setiap kata untuk menandakan pentingnya dalam dokumen dan corpus. Metode sering digunakan dalam Information Retrieval dan Text Mining.
lalu setelah itu akan melakukan fit dan transformasi ke dalam matriks, matriks tersebut adalah tfidf_matrix. Pada Gambar 5 tfidf_matrix terdapat 10000 ukuran data dan 5575 nama penulis buku. 

<br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/PWsz70x/Screenshot-2024-04-01-173228.png" alt="Screenshot-2024-04-01-173228" border="0"></a> <br>
Gambar 5 matriks <br>


Pada tabel 4 tertulis judul buku dan nama penulis, memang tidak ada yang bernilai 1.0 yang menandakan pada kolom itulah tertanda penulis dari bukunya <br>

Tabel 4. matriks dari judul buku dengan penulis - penulis buku  <br>

|book_title	|masters|paolo|lagasse|fargues|jewell|
|-----------|-------|-----|-------|-------|------|
|Unforgettable (Harlequin Intrigue, No 348)|0.0|0.0|0.0|0.0|0.0|
|All Our Yesterdays|0.0|0.0|0.0|0.0|0.0|
|Anna and the King of Siam|0.0|0.0|0.0|0.0|0.0|
|Now You See It . . .|0.0|0.0|0.0|0.0|0.0|

#### Cosine Similarity
Digunakan supaya item yang kita rekomendasikan tidak terlalu jauh dari data pusat, oleh karena itu kita butuh derajat kesamaan pada item, dalam proyek ini, buku dengan derajat kesamaan antar buku dengan cosine similarity hasilnya seperti ini <br>
array([[1., 0., 0., ..., 0., 0., 0.], <br>
       [0., 1., 0., ..., 0., 0., 0.],<br>
       [0., 0., 1., ..., 0., 0., 0.],<br>
       ...,<br>
       [0., 0., 0., ..., 1., 0., 0.],<br>
       [0., 0., 0., ..., 0., 1., 0.],<br>
       [0., 0., 0., ..., 0., 0., 1.]])<br>

Nilai K pada fungsi menandakan jumlah rekomendasi yang akan ditampilkan
Atribut argpartition berguna untuk mengambil sejumlah nilai k, dalam fungsi ini 5 tertinggi dari tingkat kesamaan yang berasal dari dataframe cosine_sim_df. Kemudian, mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini dimasukkan ke dalam variabel closest. Berikutnya, kita perlu menghapus book_title yang yang dicari agar tidak muncul dalam daftar rekomendasi. 

- Mencari Rekomendasi dari buku yang sudah dibaca 
Buku yang dibaca yaitu The Diaries of Adam and Eve berikut detailnya sesuai dengan tabel 5, ini menjadi buku yang dianggap sudah dibaca oleh user. dalam Cosine Similarity nanti akan mencari book yang mirip dengan The Diaries of Adam and Eve, sehingga perlu drop book_title The Diaries of Adam and Eve agar tidak muncul dalam daftar rekomendasi yang diberikan nanti. 

Tabel 5. Hasil rekomendasi buku yang sudah dibaca <br>

|        |**book_ISBN**|**book_title**                |**book_author**|**book_year_of_publication**|
|--------|-------------|------------------------------|---------------|------------|
|**4700**|0965881199   |The Diaries of Adam and Eve	|Mark Twain	    |1998        |

<br>
selanjutnya yaitu menampilkan buku rekomendasi. mungkin beberapa kasus ada yang menampilkan rekomendasi yang sama jadi perlu dihapus jika ada rekomendasi yang sama dengan perintah "rekomendasi.drop_duplicates()". berikut ini 5 rekomendasi buku yang ada di tabel 6, pada tabel disini ada kesamaan nama penulis bukunya berarti sistem memberikan rekomendasi buku dengan penulis yang sama dengan buku yang telah user baca. <br>


Tabel 6 lima Rekomendasi Buku<br>

|   |**book_title**|**book_author**|
|---|--------------|---------------|
|0 | ADVENTURES OF HUCKLEBERRY FINN (ENRICHED CLASS...	|Mark Twain|
|1|Adventures of Huckleberry Finn	|Mark Twain|
|2|The Complete Short Stories of Mark Twain (Bant...	|Mark Twain|
|3|Treasury of Illustrated Classics: Adventures o..| Mark Twain|
|4|A Connecticut Yankee in King Arthur's Court (D...	|Mark Twain|



###  _2. Collaborative Filtered Recommendation System_

   Collaborative Based Filtering adalah sistem rekomendasi berdasarkan pendapat suatu komunitas.
- Kelebihan pada Collaborative Based Filtering bila dibandingkan dengan Content Based Filtering adalah pengguna dapat mengeksplorasi item atau konten di luar preferensi pengguna. Pengguna pun juga dapat mendapat rekomendasi sesuai dengan kecenderungan publik yang dianalisa lewat penilaian pengguna - pengguna lainnya.
- Kekurangan pada Collaborative Based Filtering adalah pengguna kurang mendapatkan rekomendasi sesuai preferensi pribadi. Konten - konten yang diberikan oleh sistem rekomendasi lebih banyak berasal dari preferensi publik dan bukan preferensi pribadi.Pada Collaborative Based Filtering, menggunakan penilaian dari pengguna - pengguna untuk mendapatkan rekomendasi buku - buku.

pada _Collaborative Filtered Recommendation System_ menerapkan teknik collaborative filtering untuk membuat sistem rekomendasi. Teknik ini membutuhkan data rating dari user. menghasilkan rekomendasi sejumlah buku yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, kita akan mengidentifikasi buku-buku yang mirip dan belum pernah baca oleh pengguna untuk direkomendasikan. Kita akan menggunakan teknik _collaborative filtering_ untuk membuat rekomendasi ini. 

Pada tahap ini, model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

#### _Binary Crossentropy_
Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer
- Mendapatkan Rekomendasi
mendefinisikan ulang book_datase dan rating_dataset
akan mengambil user_id secara acak dari rating_dataset. Dari user_id ini kita perlu mengetahui buku - buku apa saja yang pernah dibaca dan yang belum pernah dibaca, sehingga kita hanya dapat merekomendasikan buku - buku yang belum dibaca.
hasilnya seperti gambar  7 ini <br>
<br>
Tabel 7 menampilkan rekomendasi buku buku dengan _user_ : 278418. sistem memberikan rekomendasi buku yang mungkin user juga belum membacanya dan mungkin ada kesamaan dalam suatu hal bisa dari penulis yang sama, atau mungkin rating yang bagus<br>

Tabel 7. 10 Rekomendasi Buku <br>

| | Book Title | Book Author |
|-|------------|-------------|
|0|The Funhouse |Dean R. Koontz |
|1|Fear Nothing |DEAN KOONTZ |
|2|The Pelican Brief | John Grisham |
|3|Dark Rivers of the Heart |Dean R. Koontz|
|4|Presumed Innocent | Scott Turow |
|5|Fortune's Hand | Belva Plain |
|6|Ordinary People | Judith Guest |
|7|The Wheel of Fortune | Susan Howatch |
|8|Four Past Midnight | Stephen King|
|9|Koko | Peter Straub |


### Evaluation
### _1. Content Filtered Recommendation System_
   Dalam evaluasi yang digunakan adalah precision yaitu salah satu metrik yang digunakan untuk mengukur seberapa akurat sistem rekomendasi dalam memberikan rekomendasi yang relevan kepada pengguna. membandingkan tingkat kesamaannya
metrik evaluasi 
precision: Jumlah buku yang memiliki kemiripan dalam buku yang direkomendasikan/ Jumlah buku yang direkomendasikan
penulis buku yang sudah di baca Mark Twain seperti tabel 5, dan jumlah buku yang direkomendasikan tabel 6
jadi precision : 5/5 = 100% sama


###  _2. Collaborative Filtered Recommendation System_

Nenggunakan root mean squared error (RMSE) sebagai metrics evaluation. 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/MDhMQ3V/download-19.png" alt="download-19" border="0"></a>
<br>
Gambar 6 Hasil _Training model_<br>

Perhatikanlah pada gambar 6, proses training model cukup smooth dan model konvergen pada epochs sekitar 30. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.1985 dan error pada data validasi sebesar 0.3448. 

### Kesimpulan 
1. Sistem ini mampu menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik content-based filtering agar bisa dengan mudah mencari buku yang bertipe sama dengan riwayat pemakaiannya. Dalam program ini merekomdasikannya berdasarkan kesaamaan penulis. <br/>

2. Sistem ini Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering agar orang yang baru memakai buku atau ingin membaca buku bertipe berbeda dari sebelumnya bisa mendapatkan rekomendasi yang mungkin sedang banyak digemari saat ini. <br>


Referensi Jurnal : <br>
[1]	A. Suryana, I. B. Zaki, J. Sua, G. Phua, J. Jekson, and C. Celvin, “Pentingnya Membaca Buku bagi Generasi Baru di Era Teknologi Bersama Komunitas Ayobacabatam,” Natl. Conf. Community Serv. Proj., vol. 3, pp. 715–720, 2021, [Online]. Available: https://journal.uib.ac.id/index.php/nacospro/article/view/6010




