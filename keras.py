## Veri Açıklaması: Bankanın müşterilerinin bankalarından ayrılıp ayrılmayacağı
# üzerine tahminlemesi

## Importing the libraries

## numpy modülü array çarpımları gibi yüksek işlem gücü isteyen lineer cebir
# işlemlerini bize fonksiyonel olarak sunan bir modül

## pandas modülü verimiz üzerinde işlem yapmamızı sağlayan bir modül. Veriyi
# çekme, ayırma vb. gibi işlemler için kullanılıyor.
import numpy as np
import pandas as pd

## Importing the dataset

## Verisetimizi projemize yüklüyoruz. .csv olması read_csv() fonksiyonunu
# kullandığımız için önemli.
dataset = pd.read_csv('Churn_Modelling.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 3-13'ün feature, 13'ün de class olacağı
# görülüyor. Alttaki satırda da veri setinin 3-13 arası alınıyor.
X = dataset.iloc[:, 3:13]

## 13. kolon da class olarak alınıyor.
y = dataset.iloc[:, 13]

# Encoding categorical data

## Aşağıdaki kütüphaneden LabelEncoder sınıfını çağırarak "Encoding Categorical
# Data to Numerical Data" işlemini, OneHotEncoder sınıfı ile de nominal veriler
# için gerekli ekstra dummy kolon oluşturma işlemini gerçekleştireceğiz.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

## OneHotEncoder sınfından oluşturduğumuz nesneler ile gerekli kolonlar yani
# nominal kolonlar için dummy kolon oluşturma işlemini yapıyoruz. 


#columntransformer sayesinde birden fazla heterojen kolonu birleştiriyor.
c = ColumnTransformer([("encoder", OneHotEncoder(), [1,2])], remainder = 'passthrough')
X = c.fit_transform(X)

## Feature Scalling

## Veri önişlemenin son aşamalarından olan feature sclaing yapıyoruz. Veri
# setimizde diğer kolonlara baskın çıkabilecek sayısal değerlere sahip
# kolonlar var. Bu durumu bertaraf etmek için feature scale yöntemi olan
# standartlaştırma yöntemini kullanmak adına StandardScaler sınıfını import
# ediyor ve X veri setine uyguluyoruz. y veri setine feature scale uygulayıp
# uygulamamak ise fark edici bir nokta değil ama genelde uygulamak tercih
# edilen seçim
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## Splitting the dataset into the Training set and Test set

## Tüm veri ön işleme aşamalarından sonra test-train ayrımı yapılıyor. Bunun için
# aşağıdaki kütüphanenin fonksiyonu kullanılıyor.
from sklearn.model_selection import train_test_split

## Fonksiyon 4 tane değer döndürüyor. Bunlar X yani feature kolonları için
# eğitim ve test, ile y yani class kolonu için eğitim ve test verileri.
# random_state parametresinin 1
# olması bu parametreyi kullanıp 1 yapan herkese karışık ama aynı veri
# setinin geleceğini ifade ediyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


## Artık ANN zamanı.Yapay Sinir Ağı kodlamak için keras kütüphanesini import
# ediyoruz. Bilgisayarımıza terminale 'pip install keras' yazarak kütüphaneyi
# yükleyebiliriz. YSA kodlamak için models ve layers modüllerini de import
# ediyoruz.

import keras
from keras.models import Sequential
from keras.layers import Dense #katmanları oluşturmak için

## models kütüphanesinden import ettiğimiz Sequential sınıfından nesne
# oluşturuyoruz. Bu kod ile genel YSA tanımlamasını yapıyoruz. Katman eklemekten,
# ağı eğitmeye, tahminleri ortaya çıkarmaya kadar tüm işlemleri bu classifier
# nesnesi ile yapacağız.
classifier = Sequential() #yapay sinir ağı objesi oluşturduk.

## classifier nesnesinden add metodu ile ilk katmanı ekliyoruz. add metodu içine
# Dense sınıfı ve bu sınıfın constructor yapısına gerekli parametreleri girerek
# ilk katmanımızı ekliyoruz. dense sayesinde kaç tane gizli katman olacağı
#bias yani düzeltme fonksiyonu olup olmayacağı verilebilir.
#activaiton ile hangi aktivasyon fonkiyonunu kullandığımızı vereceğiz. 
# bizim burda 11 tane bağımsız 1 tane bağımlı değişkenimiz var 
#burda şimdi units değerimizi yanş kaç tane bağımsız değişken olduğunu verecez.
#bağımsız değişkenlerimizin yarısı olacak şekilde vereceğiz. 
#bu değer genelde her problemde 6 verilir. 
#init değeri ilk değerimizin atanması için vereceğimiz değer.
#bu değerler sıfıra yakın bir değerler olmalı. 
#activasyon da kullanmak istediğimiz fonksiyonu belirticez. 
#son paremetre inputta kaç değer olduğu olacak. 

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
#uniform, agirliklarin 0 yakin random olarak verilmesini saglar

# İkinci gizli katmanımızı da aynı şekilde ekliyoruz. Bu sefer 
#  input_dim parametresi yok çünkü zaten inputları ekledik.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Son olarak çıkış katmanımızı ekliyoruz. Kodlama olarak bunu da aynı şekilde
# ekliyoruz, sadece units parametresini 1 yapıyoruz. Tahmin edeceğimiz değerler
# yani labelımız 0 ve 1'den oluşuyordu. İkili değer olduğu için tek çıkış nöronu
# yeterli oluyor.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#0 ve 1 gibi 2 sinif varken output icin sigmoid kullanmak daha avantajli, diger turlu softmax kullanmali

## YSA için eğitimde gerekli diğer hiperparametreleri belirleme zamanı. optimizer
# parametresi öğrenme fonksiyonu seçimi için, loss parametresi loss fonksiyonu
# seçimi için kullanılıyor. metrics parametresi ise hata kriterini accuracy'e göre
# belirleyeceğimiz anlamına geliyor. Tüm bunları compile metodu ile yapıyoruz.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss function da 2 deger oldugu icin binary, 2 den fazla olsaydi cross entropi olcakti.

## Artık eğitim zamanı. fit metodu ile eğitimi gerçekleştirceğiz. X_train ve
# y_train'i veriyoruz xten yyi öğren. 
#batch_size, epochs ve shuffle parametrelerine de standart
# olarak tercih edilen değerleri giriyoruz.
#epochs kaç seferde öğrenecek 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, shuffle = True)
#batch size veriyi kacarli egitecegimiz, epoch butun veriyi toplam kac kere egitecegimiz


## Predicting the Test set results

## Algoritmanın eğitimi tamamlandı. Performansını ölçmek adına test için
# ayırdığımız eğitime karışmamış verileri modele veriyoruz ve bize test setindeki
# verilerin tahminlerini yapıyor.
y_pred = classifier.predict(X_test) #bakalım buna
y_pred = (y_pred > 0.5) #0.5 den kucuk olanlari false kabul ediyor.

## Making the Confusion Matrix

## Tahminleri yaptırdıktan sonra doğruluk oranımızı görmek ve modelimizin somut
# çıktısını almak adına Confusion Matrix'i hesaplatıyoruz. Hesaplatmak için
# çağırdığımız kütüphanedeki fonksiyona görüldüğü üzere test setinin gerçek
# verilerini ve modelin tahmin ettiği verileri veriyourz. Bu adım uygulanmadan
# önce öğrencilere Confusion Matrix anlatılabilir.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
