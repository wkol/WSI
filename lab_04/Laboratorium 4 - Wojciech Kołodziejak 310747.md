## Laboratorium 4 - Wojciech Kołodziejak 310747

### Treść zadania:

Zaimplementuj algorytm SVM oraz zbadaj działanie algorytmu w zastosowaniu do zbioru danych [Wine Quality Data Set.](https://archive.ics.uci.edu/ml/datasets/wine+quality) W celu dostosowania zbioru danych do problemu klasyfikacji binarnej zdyskretyzuj zmienną objaśnianą. Pamiętaj, aby podzielić zbiór danych na zbiór trenujący oraz uczący.
Zbadaj wpływ hiperparametrów na działanie implementowanego algorytmu. W badaniach rozważ dwie różne funkcje jądrowe poznane na wykładzie.

### Opis implementacji:

##### Metoda SVM

Metoda SVM jest klasyfikatorem używającym hiperpłaszczyzn do dzielenia zbioru punktów na klasy. W zadaniu została zrealizowana klasyfikacja binarna w zastosowaniu do zbioru danych win. W celu zdyskretyzowania zmiennej objaśnionej będącej jakością poszczególnych win został użyty prosty algorytm dzielący wartości na dwie klasy:

* wino dobre (1) - w przypadku gdy jakość wina jest większa niż  5
* wino słabe (-1) - w przypadku gdy jakość wina jest mniejsza równa niż 5

Następnie algorytm klasyfikacji SVM znajduje hiperpłaszczyznę o jak największym marginesie przy użyciu wektorów nośnych. Aby wyznaczyć tą płaszczyznę należy znaleźć maksymalny argument funkcji będącej szerokością marginesu:
$$ {LARGE}
\LARGE
\begin{aligned}
min \ \frac{1}{2} \norm{w}^2\\
\end{aligned}
$$


Przy założeniu:
$$
\large
\begin{aligned}
\sum_{i=1}^n y_i(w\cdot x_i + b) - 1 \geq 0
\end{aligned}
$$
Korzystając z mnożników Lagrange'a otrzymujemy następną postać:
$$
\LARGE
L = \frac{1}{2} \norm{w}^2 - \sum_{i=1}^n \alpha_i[y_i(w \cdot x_i + b) - 1]
$$




W przypadku nieliniowego SVM po przekształceniach otrzymujemy funkcje do maksymalizacji w postaci:
$$
\large
\begin{aligned}
L = \sum_{i=1}^n \alpha_{i} - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^ny_iy_j\alpha_i\alpha_j(\phi(x_i) \ \dot{} \ \phi(x_j))
\end{aligned}
$$
Przy założeniach:
$$
\large
\begin{aligned}
\sum_{i=1}^n \alpha_i y_i = 0 \ \textrm{oraz } 0 \leq \alpha_i < C
\end{aligned}
$$


Jednak zamiast przekształcać argumenty do wyższej przestrzeni skorzystamy z funkcji jądrowych które zwracają jedynie skalar co pozwala zaoszczędzić zasoby. W implementacji użyte zostały następujące jądra:

* rbf - określone wzorem $$ \large exp(-\gamma*\norm{x_i - x_j}^2)$$ gdzie parametr $$ \large \gamma $$ podaje użytkownik
* wielomianowe - określone wzorem $$\large \gamma((x_i\cdot x_j) + r)^{deg} $$   gdzie parametry $$\large \gamma, r, deg $$ podawane są przez użytkownika

Po zamianie przekształceń na funkcje jądrowe otrzymujemy finalną funkcję którą musimy zmaksymalizować przy tych samych założeniach:
$$
\LARGE
\begin{aligned}
L = \sum_{i=1}^n \alpha_{i} - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^ny_iy_j\alpha_i\alpha_jk(x_i, x_j)
\end{aligned}
$$
Po wyznaczeniu parametrów $$ \large \alpha_i $$ dla których funkcja osiąga maksimum możemy wyliczyć równanie płaszczyzny, a następnie klasyfikować nowe punkty zależnie, po której stronie hiperpłaszczyzny leżą

##### Implementacja w języku python

W implementacji zostały użyte następujące biblioteki:

* numpy - operacje na macierzach
* cvxopt - solver optymalizujący
* matplotlib - wykres
* sklearn - podział zbioru na uczący i walidujący oraz wyliczanie dokładności 

W zadaniu została użyta biblioteka cvxopt, która posiada solver qp przyjmujący problemy w następującej formie:
$$
\LARGE
\begin{aligned}
\begin{aligned}
&\textrm{minimalizuj} & \frac{1}{2}x^TPx+q^Tx\\
\\
&\textrm{przy założeniach} & Gx\leq h\\
& & Ax=b
\end{aligned}
\end{aligned}
$$
W celu poprawnego działania solvera nasz problem musiał zostać przekonwertowany do poprawnej formy.

1. Przekształcić równanie L aby pasowało do optymalizacji $$ \large L = \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^ny_iy_j\alpha_i\alpha_jk(x_i, x_j) - \sum_{i=1}^n \alpha_{i} $$
2. Przyjąć zmienną do zoptymalizowania jako $$\large \alpha$$
3. Wyliczyć poszczególne parametry solvera:
   * macierz P - możemy zauważyć że wyrażenie $$ \large \frac{1}{2}x^TPx $$ rozwija się do macierzy kwadratowej o n wierszach gdzie każde pole jest równe $$ \large x_ix_j\cdot P_{ij}$$ gdzie i - wiersz, j - kolumna. Podstawiając $$ x = \alpha $$  i wyliczając P otrzymujemy $$ \large P_{ij} = y_iy_jk(x_i, x_j) $$
   * macierz q - rozwija się do wyrażenia $$ \large \sum_{i=1}^n q_i\alpha_i $$ w związku z czym aby pasowało do naszego równania przyjmujemy za macierz wypełnioną -1
   * macierz G - w naszym przypadku założeniem jest $$ \large 0 \leq \alpha_i < C $$ w związku z czym macierz G będzie złączeniem pionowym dwóch macierzy - jednej diagonalnej z wartościami 1, drugiej diagonalnej z wartościami -1
   * macierz h - aby w naszym przypadku założenie $$ \large 0 \leq \alpha_i < C $$ było prawdziwe macierz h musi być złączeniem poziomym dwóch macierzy - jednej  macierzy wypełnionej zerami, drugiej wypełnionej wartościami C
   * macierz A - drugie z założeń w naszym przypadku wynosi: $$ \large \sum_{i=1}^n \alpha_i y_i = 0 $$ gdy za $$\large x$$ podstawimy $$\large \alpha$$ dojdziemy do wniosku, że macierz A musi być równa wektorowi klas wszystkich punktów ze zbioru treningowego 
   * macierz b - w naszym przypadku jest równa 0

 4. Mając dane potrzebne do wyliczenia przez solver poszczególnych alf wybieramy pierwszy wektor nośny i wyliczamy dla niego parametr b z równania:
    $$
    \LARGE
    \begin{aligned}
    b=y_{firstSV} - \sum_{i=1}^n \lambda_i y_i (x_i\cdot x_{firstSV})
    \end{aligned}
    $$

 5. Gdy obliczymy wszystkie potrzebne dane możemy zająć się predykcją klasy jest ona realizowana przez funkcje która sprawdza znak wyrażenia:
    $$
    \Large
    \sum_{i=1}^n \alpha_i y_i k(x_i, u)
    $$
    Gdzie u to punkt który chcemy sklasyfikować

    Jeśli wartość tego wyrażenia jest większa równa od zera przypisujemy punktowi klasę 1 (wino dobre) w przeciwnym wypadku klasę -1 (wino słabe)

### Opis eksperymentów numerycznych:

W celu zbadania wpływu hiperparametrów na działanie algorytmu SVM, dane znajdujące się w `winequality-red.csv` zostały losowo podzielone na zestawy trenujący i testujący w proporcji 4:1 oraz zmienna `quality` została zdyskretyzowana na dwie klasy(-1, 1). Następnie dla zarówno funkcji jądrowej rbf oraz wielomianowej zostały przeprowadzone eksperymenty polegające na wielokrotnym uruchomieniu algorytmu dla różnych parametrów co pozwala na ustalenie najlepszych wartości takich jak: 

* C - parametr regulujący częstotliwość błędnego zaklasyfikowania 
* gamma - parametr służący do obliczania wartości funkcji jądrowej 

Przy jądrze wielomianowym zostały użyte parametry deg = 2 oraz r = 3.

#### Wyniki eksperymentu

Wyniki dla jądra wielomianowego wyglądają następująco:

![plot_polynomial](/home/wojtek/Documents/Studia/sem3/plot_polynomial.png)

Widzimy że dla jądra wielomianowego zachowanie algorytmu jest dobre niezależnie od przyjętych wartości parametru gamm

Dla jądra rbf wykres wpływu parametru gamma i C na dokładność wygląda następująco:

![plot_rbf](/home/wojtek/Documents/Studia/sem3/WSI/lab_04/plot_rbf.png)

### Wnioski:

Z przeprowadzonych eksperymentów możemy dojść do wniosku że algorytm przy odpowiednio dobranych parametrach działa "dobrze" (porównywalne wyniki z użyciem pakietu `sklearn.svm`), jednak nadal dobór parametrów odgrywa ważną rolę. Dostrajając oprócz parametrów C i gamma również inne parametry jak np. wielkość zbioru uczącego możemy osiągnąć nawet lepsze wyniki niż ukazane na wykresach. W większości przypadków otrzymujemy dokładność około 70% (dla win czerwonych). 

