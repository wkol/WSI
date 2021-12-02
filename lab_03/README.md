### Gra w kółko i krzyżyk z wykorzystaniem algorytmu minimax oraz heurestyki 

Gra pozwala na rozgrywkę na kwadratowej planszy o podanej przez użytkownika długości. Użytkownik może wybrać rodzaj graczy:
* gracz AI wykorzystujący algorytm minimax o maksymalnej głębkości podawanej jako argument z linii poleceń. W celu skrócenia czasu obliczania ruchu jest wykorzystywana podstawowa funkcja obliczająca wartość heurystyczną planszy, dzięki czemu algorytm przy mniejszych zadanych głębokościach jest w stanie grać nadal optymalnie
* gracz użytkownik podający wspołrzędne swojego ruchu przy użyciu konsoli

#### Uruchomienie
Aby uruchomić program należy uruchomić plik `main.py` używając interpretera python: np. `python3 main.py`. Opcjonalnie można dodać parametr ustawiający głębokość poszukiwania gracza AI, używając `python3 main.py --depth <ZADANA GŁĘBOKOŚĆ>`. Program po uruchomieniu pyta użytkownika przy użyciu konsoli o wielkość planszy, a następnie o rodzaj graczy. Po zakończonej rozgrywce program pyta o rozpoczęcie następnej rozgrywki. Jeśli zostanie wybrana opcja Y, program wyczyści planszę i zacznie kolejną rozgrykę.