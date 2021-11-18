## Laboratorium 2 - Wojciech Kołodziejak 310747

### Treść zadania:

Zaimplementuj strategię ewolucyjną typu ($$\mu/\mu,\lambda $$)-ES, w której rozważysz dwa mechanizmy adaptacji zasięgu mutacji $$\large\sigma$$

- samoadaptację (ang. Self-Adaptation, SA)
- metodę logarytmiczno-gaussowską (ang. Log-Normal Mutation Rule, LMR).

Przy czym metoda logarytmiczno-gaussowska dana jest następującym równaniem:
$$
\large
\begin{aligned}
\sigma^{t+1} = \sigma^{t}e^{\tau N(0, 1)}
\end{aligned}
$$
natomiast samoadaptacja przebiega w następujący sposób:
$$ {aligned}
\large
  \begin{aligned}
  \xi^{t}_{i} &= \tau\mathcal{N}_{i}(0, 1)\\
  \pmb{z}^{t}_{i} &= \mathcal{N}_{i}(0, \pmb{I}_{D}) \\
  \sigma^{t}_{i} &= \sigma^{t}exp\{\xi^{t}_{i}\} \\
  x^{t}_{i} &= \pmb{x}^{t} + \sigma^{t}_{i}\pmb{z}^{t}_{i},\; i \in
  1\dots\lambda
  \end{aligned}
$$


gdzie $$\mathcal{N}_{i}(0, \pmb{I}_{D})$$ oznacza próbę losową z D-wymiarowego rozkładu normalnego o wartości oczekiwanej równej 0 oraz jednostkowej macierzy kowariancji. Następnie zbadaj zbieżność obu metod w zależności od rozmiaru populacji potomnej oraz początkowej wartości zasięgu mutacji. Do badań wykorzystaj funkcję sferyczną oraz funkcję o następującej postaci:

$$
\large
  q(\pmb{x}) = [(\|\pmb{x}\|^{2} - D)^{2}]^{1/8} + D^{-1}(\frac{1}{2}\|x\|^{2}
  + \sum^{D}_{i = 1}x_{i}) + \frac{1}{2},\; \pmb{x} \in [-100, 100]^{D}
$$
W badaniach przyjmij, że:

- wagi w operatorze rekombinacji wynoszą $$\forall_{i
        \in 1\dots\mu} w_{i} = \mu^{-1}$$
- parametr $$\tau$$ jest proporcjonalny do $$1/\sqrt{D}$$
- rozkład normalny parametryzowany jest jednostkową wariancją lub macierzą kowariancji
- wymiarowość wynosi 10.

<div style="page-break-after: always;"></div>

### Opis implementacji:

W implementacji zostały użyte następujące biblioteki:

- NumPy - macierze i działania na nich
- matplotlib - sporządzanie wykresów

#### Implementacja algorytmu strategii ewolucyjnej typu  ($$\mu/\mu,\lambda $$)-ES:

```python
def mi_lambda_es(starting_point: np.ndarray, fun: Callable, rng: np.random.Generator, mi_param: int, lambda_param: int, start_sigma: float, tau: float, max_iters: int = 2000, adapt: bool = True) -> Tuple[np.ndarray, List[float]]:
        population = Population(Individual(starting_point, start_sigma), mi_param, lambda_param, rng, fun)
        population.centroid.calculate_fitness(fun)
        y_vals = []
        iters = 0
        while iters < max_iters:
            y_vals.append(population.centroid.fitness)
            population.population = population.recombine()
            if(adapt):
                population.update_self_adapt_sigma(tau)
            else:
                population.update_sigma(tau)
            population.mutation()
            population.calculate_fitness()
            population.population = population.sort_population()
            population.population = population.succesion()
            population.update_centroid()
            iters += 1
        return population.centroid.params, np.array(y_vals)
```

Algorytm strategii ewolucyjnej typu ($$\mu/\mu,\lambda$$)-ES działa w następujący sposób: 

1. Przyjęcie potrzebnych argumentów parametryzujących działanie algorytmu w tym:
   * punkt startowy [x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>...x<sub>n-1</sub>]
   * zadaną funkcje do znalezienia minimum
   * generator służący do generowania losowych liczb
   * parametr $$\mu$$ oznaczający liczebność populacji rodziców
   * parametr $$\lambda$$ oznaczający liczebność generowanych rekombinantów
   * początkową sigmę
   * parametr $$\tau$$ biorący udział w mechanizmie adaptacji kolejnych sigm
   * liczbę iteracji przewidzianych dla algorytmu
   * mechanizm adaptacji (LMR lub SA)

2. Utworzenie z punktu początkowego centroidu o konkretnych wartościach argumentów oraz sigmie
3. Rekombinacja populacji polegająca na wygenerowaniu $$\lambda$$ nowych osobników mających argumenty centroidu
4. Zaadaptowanie sigmy zależnie od parametru `adapt`:
   * w przypadku LMR jedna sigma jest adaptowana dla całej populacji
   * w przypadku SA sigma jest adaptowana dla każdego osobnika

5. Mutacja polegająca na dodaniu szumu do argumentów każdego z osobników
6. Obliczenie funkcji celu dla każdego z osobników
7. Posortowanie osobników rosnąco ze względu na wartość funkcji celu
8. Wybranie $$\mu$$ najlepszych osobników z populacji (tj. z najniższymi wartościami funkcji celu)
9. Wygenerowanie nowego centroidu
10. Sprawdzenie warunku iteracji. Gdy nie jest spełniony algorytm wraca się do punktu 3, w przeciwnym razie zwracane zostają parametry centroidu oraz kolejne wartości funkcji w kolejnych iteracjach

Wybranie poszczególnych parametrów ma duże znaczenie w szczególności dla metody LMR, zależnie od wielkości rodziców oraz liczebności potomków algorytm wykonuje się w różnym czasie oraz dociera do wyników rożnej dokładności. Również dla wariantu LME duży wpływ ma wartość początkowej sigmy. Zależnie od niej algorytm może momentami bardzo oddalać się od minimum.

### Opis eksperymentów numerycznych:

W celu zbadania działania algorytmu strategii ewolucyjnej dla każdego z mechanizmów adaptacji sigmy zostały przeprowadzone eksperymenty obrazujące zbieżność lub rozbieżność danej metody. W związku z tym zostały wybrane różne zestawy parametrów wejściowych funkcji:
$$
\large
\large
(\mu, \lambda) \in \{(4, 20), (20, 100), (100, 200)\} \\
\large
\sigma_{start} \in \{0.01, 0.1, 1\}\\
$$
Kolejne wartości funkcji w danych iteracjach zostały uśrednione z 5 wywołań funkcji, każde z innym ziarnem, co pozwala uzyskać bardziej wiarygodne i niezależne wartości. Również każdy z eksperymentów został przeprowadzony zarówno dla podanej w zadaniu funkcji celu oraz dla funkcji sferycznej określonej wzorem $$f(\textbf{x}) = \sum^{10}_{i = 1}x^{2}_{i}$$.

### Wyniki eksperymentów:

* Funkcja celu $$ \large q(\pmb{x})$$:

  * $$\large (\mu, \lambda) = (4, 20)$$ <img src="/home/wojtek/Documents/WSI/lab_02/chart/fig3.png" alt="fig3.png" style="zoom:40%"/>

    

  * $$\large (\mu,\lambda) = (20, 100)$$<img src = "/home/wojtek/Documents/WSI/lab_02/chart/fig4.png" alt="fig4" style="zoom:40%"/>

  * $$\large (\mu,\lambda) = (100, 200)$$<img src = "/home/wojtek/Documents/WSI/lab_02/chart/fig5.png" alt="fig5" style="zoom:40%"/>

* Funkcja sferyczna  $$\large f(\textbf{x})$$

  * $$\large (\mu,\lambda) = (4, 20)$$

    <img src="/home/wojtek/Documents/WSI/lab_02/chart/fig0.png" alt="fig0" style="zoom:40%;" />

    $$\large (\mu,\lambda) = (20, 100)$$<img src="/home/wojtek/Documents/WSI/lab_02/chart/fig1.png" alt="fig1.png" style="zoom:40%"/>

  * $$\large (\mu,\lambda) = (100, 200)$$ <img src="/home/wojtek/Documents/WSI/lab_02/chart/fig2.png" alt="fig2.png" style="zoom:40%"/>
  
    

#### Porównania czasowe:

* Średni czas wykonywania algorytmu strategii ewolucyjnej ($$\mu/\mu,\lambda$$):
  * z metodą adaptacji sigmy logarytmiczno-gaussowską: 4.0749 s
  * z metodą adaptacji sigmy samoadaptacyjną: 4.1680 s

### Wnioski:

Z przeprowadzonych eksperymentów wynika, że metoda z samoadaptacją radzi sobie o wiele lepiej niż metoda z logarytmiczno-gaussowskim mechanizmem adaptacji. Niezależnie od parametrów wejściowych metoda SA znajdowała średnio punkt zerowy z o wiele większą dokładnością oraz wartość funkcji nie rozbiegała w czasie iteracji. W przypadku metody LMR działanie algorytmu było mocno zależne od parametrów oraz losowości. Funkcja w niektórych, pojedynczych, nieuśrednionych przypadkach zbiegała do minimum, jednak w większości przypadków w trakcie iteracji rozbiegała do bardzo odległych wartości na co wpływ mógł mieć mechanizm przypisywania jednej wartości sigmy dla całej populacji. Gdy wybrana zostaje sigma o większej wartości i następnie przypisywana do każdego osobnika, argumenty funkcji każdego z nich zostają bardzo zmienione, co skutkuje oddalaniem się od minimum funkcji.

