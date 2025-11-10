# Simulador de Propagació de Virus - EGRA

## Descripció
Es requereix construir un simulador de propagació d’un virus en un espai tancat determinat, en aquest cas, dins de l’Escola d’Enginyeria, en un dia lectiu habitual. La idea principal és visualitzar l’evolució dels infectats entre els estudiants i professorat, analitzar com el moviment de persones i la distància entre elles influeixen en el número d’afectats, i avaluar l’expansió del virus. Es definiran les aules i es modelarà comportament dels personatges en l’escena tenint en compte l’horari lectiu (hora de sortida, hora d’entrada, descansos,etc.). Finalment, s’extraurà tota la informació relacionada amb la malaltia propagada durant el temps definit. 

## Requisits
- Python 3.8 o superior
- Biblioteques necessàries:
  - pygame
  - moderngl
  - PyGLM
  - numpy

### Instal·lació de dependències
```bash
pip install pygame moderngl PyGLM numpy
```

## Estructura del Projecte
```
DEMO/
├── DEMO.py              # Motor gràfic principal
├── facultat.py          # Sistema de sales i waypoints
├── person.py            # Classe persona
├── virus.py             # Lògica de contagi
├── camera.py            # Control de càmera
├── escenario.py         # Renderització de l'escenari
├── marker.py            # Marcador 3D
└── data/salas/          # Definició de sales (JSON)
    ├── aula1.json
    ├── aula2.json
    ├── aula3.json
    └── pasillo.json
```

## Execució
Obriu un terminal a la carpeta del projecte i executeu:

```bash
cd DEMO
python DEMO.py
```

## Controls

### Càmera
- **WASD**: Moure la càmera
- **Espai**: Pujar
- **Ctrl**: Baixar
- **Shift + moviment**: Sprint
- **Ratolí**: Rotar la vista
- **TAB**: Alliberar/capturar el cursor del ratolí

### Simulació
- **P**: Iniciar/Pausar la simulació
- **R**: Reiniciar la simulació (esborrar totes les persones)
- **ESC**: Sortir de l'aplicació

### Marcador (eina opcional)
- **Fletxes direccionals**: Moure marcador en els eixos X/Z
- **P/L**: Pujar/Baixar marcador en l'eix Y

## Funcionament de la Simulació

1. **Inici**: En prémer la tecla **P**, comença el spawn automàtic de persones al passadís
2. **Moviment**: Les persones es mouen automàticament seguint el seu horari assignat (`schedule`)
3. **Contagi**: 
   - Les persones infectades es marquen amb un **anell vermell** 3D
   - El contagi es comprova cada `tick_duration` segons
   - La probabilitat de contagi depèn de la distància entre persones
   - La primera persona que entra a cada aula s'infecta automàticament
4. **Visualització**: Els infectats tenen un anell visual i partícules que indiquen el seu estat

## Paràmetres de Configuració

Podeu ajustar els paràmetres del virus editant el fitxer `DEMO.py`:

```python
self.tick_duration = 0.2           # Temps entre comprovacions (segons)
self.infection_probability = 0.2   # Probabilitat de contagi (0.0 - 1.0)
self.infection_distance = 0.9      # Distància màxima de contagi (unitats)
self.intervalo_spawn = 4.0         # Temps entre spawn de persones (segons)
self.max_people = 50               # Nombre màxim de persones simultànies
```

## Configuració de Sales

Les sales es defineixen mitjançant fitxers JSON a `DEMO/data/salas/`. Cada fitxer conté:

- **tipo**: Tipus de sala ("clase", "pasillo", etc.)
- **pos**: Diccionari de posicions dels waypoints `{id: [x, y, z]}`
- **con**: Diccionari de connexions entre waypoints `{id: [ids_veïns]}`
- **entrada/salida**: IDs dels waypoints que fan de porta
- **asientos**: (només aules) IDs dels seients disponibles

### Exemple d'estructura JSON
```json
{
  "tipo": "clase",
  "pos": {
    "0": [x, y, z],
    "1": [x, y, z]
  },
  "con": {
    "0": [1, 2],
    "1": [0, 3]
  },
  "entrada": 0,
  "salida": 5,
  "asientos": [10, 11, 12]
}
```

## Informació Tècnica

### Sistema de Pathfinding
Les persones utilitzen l'algoritme A* per navegar entre waypoints de diferents sales seguint el seu horari.

### Sistema de Contagi
- Es comprova la distància entre cada persona infectada i les persones sanes
- Si la distància és menor que `infection_distance` durant un tick, hi ha probabilitat de contagi
- La probabilitat es controla amb el paràmetre `infection_probability`

### Renderització
- Motor OpenGL 3.3 amb ModernGL
- Shaders Phong per a il·luminació realista
- Càmera lliure amb controls FPS

## Notes Importants
- La simulació comença pausada. Cal prémer **P** per iniciar-la
- Els FPS es mostren a la barra de títol de la finestra
- Les persones s'esborren automàticament en prémer **R**
- Assegureu-vos que els fitxers OBJ i JSON estan a les rutes correctes

## Autors
Projecte EGRA - Propagació de Virus
Adrià Fraile, Adrian Díaz, Amina Aasifar, Cristian Rey i Joan Colillas 
Escola d'Enginyeria

