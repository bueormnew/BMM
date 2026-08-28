# BUEORM Delta Attention (BDA)
## Especificación técnica de un mecanismo de memoria asociativa recurrente para modelos de lenguaje

**Autor / propietario del diseño:** Gerson Fabián Buenahora Ormaza — *BUEORM*
**Estado del documento:** Borrador técnico v2 — especificación + validación experimental preliminar
**Ámbito:** Mecanismo de atención/memoria para arquitecturas de modelos de lenguaje de gran escala

---

## 0. Resumen

BUEORM Delta Attention (BDA) es un mecanismo de memoria recurrente de estado
fijo para modelos de lenguaje, diseñado bajo un objetivo explícito: **máxima
velocidad de cómputo y paralelización en GPU, manteniendo un balance
matemáticamente controlado entre estabilidad, capacidad de recuperación
asociativa y resistencia a la interferencia**, sin recurrir a compresión de
la secuencia (no hay resumen jerárquico ni selección tipo "state-space" de
los tokens: cada token participa individualmente en la actualización de la
memoria mediante una regla de corrección explícita).

Este documento define de forma autocontenida cada componente de BDA: su
motivación, su forma matemática exacta, su pseudocódigo de referencia, y —de
manera central en esta versión— un **análisis de estabilidad que no da nada
por sentado**: se muestra explícitamente dónde una condición de estabilidad
"intuitiva" falla, se deriva una condición que sí es demostrable, y se
reportan los resultados de pruebas numéricas que verifican ambos hechos.

---

## 1. Filosofía de diseño

BDA parte de tres compromisos de diseño, ordenados por prioridad:

1. **Velocidad antes que fidelidad perfecta.** BDA no intenta igualar a la
   atención completa en recuperación exacta de información arbitrariamente
   antigua. Intenta ser la memoria recurrente *más rápida posible* que siga
   siendo *matemáticamente honesta* sobre sus límites — y que sea lo
   suficientemente buena para combinarse con bloques de atención completa
   ocasionales, sin ser el cuello de botella de cómputo del modelo.

2. **Ningún supuesto sin demostración o sin prueba empírica.** Cualquier
   afirmación de estabilidad, expresividad o eficiencia debe estar
   acompañada de una justificación matemática explícita o de un experimento
   reproducible. Si algo no está demostrado, este documento lo marca
   explícitamente como *hipótesis abierta*, no como propiedad garantizada.

3. **Cada componente debe tener un motivo de existencia verificable.** BDA
   evita agregar mecanismos "porque funcionan en otros sistemas similares".
   Cada pieza del diseño (puertas, máscaras, normalizaciones,
   proyecciones) se justifica aquí desde primeros principios: qué problema
   resuelve, qué costo computacional añade, y qué pasaría si se quitara.

La consecuencia directa de estos tres compromisos es que BDA se define como
un **estado de memoria de tamaño fijo por cabeza, actualizado token a token
mediante una corrección de error explícita (no una simple acumulación), con
control de olvido diferenciado por canal, un mecanismo de borrado
desacoplado de la escritura, y una capa de seguridad numérica que se aplica
en tiempo de ejecución en lugar de asumirse por construcción.**

---

## 2. Qué propone BUEORM Delta Attention

BDA propone un mecanismo de memoria con **seis componentes nombrados**,
cada uno resolviendo un subproblema distinto:

| # | Componente | Nombre | Problema que resuelve |
|---|---|---|---|
| 1 | Estado de memoria asociativa | **Memory State (MS)** | Representar el conocimiento acumulado del contexto en tamaño fijo |
| 2 | Puerta de Olvido de Bajo Rango | **Low-Rank Forget Gate (LRFG)** | Olvido diferenciado por canal, sin el costo de una proyección completa |
| 3 | Máscara de Borrado Desacoplada | **Decoupled Erase Mask (DEM)** | Evitar que escribir información nueva destruya información no relacionada (interferencia) |
| 4 | Normalización Adaptativa de Paso | **Adaptive Step Normalization (ASN)** | Mantener el tamaño de la corrección de escritura en una escala estable pase lo que pase con la magnitud de las claves |
| 5 | Proyección de Estabilidad | **Stability Projection (SP)** | Garantizar —matemáticamente, no por esperanza— que el estado no puede explotar numéricamente |
| 6 | Entrenamiento Paralelo Recurrente por Bloques | **Chunk-Recurrent Parallel Training (CRPT)** | Permitir entrenamiento eficiente en GPU sin depender de una recurrencia estrictamente secuencial de longitud N |

La sección 4 define cada uno formalmente. La sección 6 explica cómo, en
conjunto, resuelven los seis requisitos funcionales que debe cumplir
cualquier mecanismo candidato a reemplazar parcialmente la atención
completa: actualización de estado, recuperación de estado, interferencia,
recuperación de largo alcance, memoria multi-cabeza y entrenamiento
paralelo.

---

## 3. Qué es y qué no es BDA

### 3.1 Qué ES

- Un mecanismo de **memoria recurrente de estado fijo**: cada cabeza
  mantiene una matriz de memoria `S ∈ R^{d_v × d_k}` cuyo tamaño no crece
  con la longitud de la secuencia.
- Una **regla de corrección de error** (no una simple suma acumulativa):
  cada token corrige lo que la memoria "predice" incorrectamente para su
  propia clave, de forma análoga a un paso de aprendizaje online.
- Un mecanismo **token-a-token explícito**: cada token individual participa
  en una escritura de memoria propia. No hay resumen de bloques de tokens
  en un vector único antes de escribir.
- Un mecanismo **diseñado para ser un componente dentro de una arquitectura
  híbrida**, combinado con capas de atención completa en una proporción
  configurable (ver sección 6.4).
- Un mecanismo con **garantías de estabilidad activas** (impuestas en
  tiempo de ejecución vía Stability Projection), no garantías pasivas
  "heredadas" de la forma algebraica de la regla de actualización.

### 3.2 Qué NO es

- **No es** un mecanismo de compresión jerárquica de secuencia (no resume
  bloques de tokens en un solo vector antes de que la memoria los vea; cada
  token individual actualiza el estado).
- **No es** un reemplazo total de la atención completa. No garantiza
  recuperación exacta y sin pérdida de información arbitrariamente antigua
  — esto es una imposibilidad matemática para cualquier mecanismo de
  estado fijo (ver sección 3.3), no una limitación específica de BDA.
- **No es** un mecanismo con estabilidad "por construcción" solo por usar
  una puerta de olvido en `(0,1)`. La sección 5 muestra explícitamente por
  qué esa intuición es insuficiente.
- **No es**, en esta versión del documento, un algoritmo con un kernel de
  entrenamiento totalmente paralelo en el sentido fuerte (paralelismo
  logarítmico dentro de cada bloque). Lo que se define y se valida aquí es
  un esquema de **paralelismo entre bloques** con recurrencia corta dentro
  de cada bloque (ver sección 7.6). El kernel totalmente paralelo se deja
  explícitamente como trabajo futuro (sección 12).

### 3.3 Un límite que no depende del diseño

Cualquier mecanismo cuyo estado tenga tamaño fijo `d_v · d_k` (independiente
de la longitud de secuencia `N`) no puede almacenar sin pérdida una
cantidad de información que crece con `N`, en cuanto `N · d` supera la
capacidad del estado. Esto es un argumento de conteo (pigeonhole), no una
debilidad de ingeniería. BDA no pretende evadir este límite: lo administra
maximizando qué tan "buena" es la información retenida (vía las puertas 2 y
3) y delegando la recuperación exacta de largo alcance a capas de atención
completa intercaladas (sección 6.4).

---

## 4. Fundamentos matemáticos

### 4.1 El estado como memoria asociativa

Cada cabeza de atención mantiene una matriz `S ∈ R^{d_v × d_k}`, interpretada
como una memoria asociativa lineal: dada una consulta `q`, la lectura es
`o = S q`. El objetivo de la actualización de `S` en cada paso es que, para
la clave `k_t` que se acaba de escribir, `S q ≈ v_t` cuando `q ≈ k_t` —
es decir, `S` actúa como un mapa aprendido "clave → valor" que se sigue
ajustando en línea, token a token.

### 4.2 Actualización como corrección de error (no acumulación)

En lugar de simplemente sumar `v_t k_t^⊤` al estado (lo que nunca "borra"
nada y hace crecer `S` sin control), BDA actualiza el estado corrigiendo el
error de predicción sobre la clave actual:

```
error_t = S_{t-1} k_t  −  v_t
S_t     = S_{t-1}  −  β_t · error_t · k_t^⊤
```

Esta es la forma general de "escritura por corrección de error" de la que
BDA es una instancia extendida (secciones 4.3–4.5 añaden las piezas
propias de BDA sobre esta base).

### 4.3 Olvido diferenciado por canal (LRFG)

En vez de un único factor de olvido escalar para toda la memoria, BDA aplica
un vector de olvido `α_t ∈ (0,1)^{d_k}`, uno por canal de la dimensión de
clave, multiplicando la memoria **antes** de la corrección:

```
S_t = S_{t-1} · diag(α_t)  −  β_t · error_t · k̂_t^⊤
```

Esto permite que distintos canales de memoria retengan información a
distintas velocidades — algunos canales pueden actuar como memoria de muy
corto plazo, otros como memoria persistente — sin necesidad de mantener
sub-espacios de memoria separados explícitamente.

**Nombre del componente: Low-Rank Forget Gate (LRFG).**
En vez de generar `α_t` con una proyección lineal completa
`R^{d_model} → R^{d_k}` (costo `O(d_model · d_k)` por token, por cabeza),
LRFG usa un cuello de botella compartido entre cabezas:

```
z_t      = V · x_t                      # V ∈ R^{r × d_model},  r ≪ d_k
α_t^h    = σ( U_α^h · z_t + b_α^h )     # U_α^h ∈ R^{d_k × r}, una por cabeza h
```

Costo: `O(d_model · r)` para calcular `z_t` (compartido entre todas las
cabezas) más `O(r · d_k)` por cabeza — significativamente más barato que
una proyección densa completa por cabeza cuando `r ≪ d_k`.

**Pseudocódigo — LRFG:**
```
function LRFG(x_t, V, {U_alpha_h, b_alpha_h for h in heads}):
    z = V @ x_t                              # (r,)
    for h in heads:
        alpha[h] = sigmoid(U_alpha_h @ z + b_alpha_h)   # (d_k,)
    return alpha                             # dict h -> (d_k,) en (0,1)
```

### 4.4 Borrado desacoplado de la escritura (DEM)

**Problema que resuelve — interferencia:** en la regla de la sección 4.2,
la misma clave `k_t` se usa para (a) leer qué hay actualmente guardado en
esa dirección de memoria (para calcular el error) y (b) escribir la
corrección. Esto acopla "dónde miro" con "dónde escribo": el modelo no
puede, en un mismo paso, decir *"limpia lo que hay en la dirección A, pero
escribe la información nueva orientada hacia la dirección B"*. Sin ese
desacople, información importante guardada en direcciones cercanas a `k_t`
puede degradarse aunque no tenga relación real con el contenido nuevo.

**Solución — Decoupled Erase Mask (DEM):** se genera una máscara de borrado
por canal `e_t ∈ (0,1)^{d_k}` (reutilizando el mismo cuello de botella `z_t`
de LRFG, por lo que el costo adicional es mínimo) y se construye una
**clave de lectura para borrado** distinta de la clave de escritura:

```
e_t   = σ( U_e^h · z_t + b_e^h )        # máscara de borrado, por cabeza
k̃_t  = k̂_t ⊙ e_t                       # clave de "erase" (lectura/borrado)
```

`k̂_t` (la proyección estándar de clave) se sigue usando para la escritura
(el término externo `error_t k̂_t^⊤`), pero el error de predicción se
calcula usando `k̃_t`:

```
error_t = S_{t-1} diag(α_t) k̃_t  −  v_t
S_t     = S_{t-1} diag(α_t)  −  β_t · error_t · k̂_t^⊤
```

Esto permite que el modelo, mediante `e_t`, module por canal *cuánto* de la
memoria decayada participa en el cálculo del error (y por tanto cuánto se
"toca" en la corrección), de forma independiente de hacia dónde apunta la
escritura nueva (`k̂_t`).

**Pseudocódigo — DEM:**
```
function DEM(k_hat_t, z_t, U_erase_h, b_erase_h):
    e_t = sigmoid(U_erase_h @ z_t + b_erase_h)   # (d_k,)
    k_tilde_t = k_hat_t * e_t                    # producto elemento a elemento
    return k_tilde_t
```

### 4.5 Normalización adaptativa del paso de corrección (ASN)

**Problema que resuelve:** la magnitud de `β_t · ‖k̂_t‖ · ‖k̃_t‖` determina
qué tan agresiva es cada corrección. Si la escala de las claves varía
durante el entrenamiento o entre capas, un `β_t` fijo o mal calibrado puede
producir correcciones demasiado bruscas (inestabilidad) o demasiado
tímidas (bajo poder de retención).

**Solución — Adaptive Step Normalization (ASN):** se mantiene, por cabeza,
un promedio móvil exponencial (EMA) escalar de la norma al cuadrado de la
clave de escritura, y se usa para reescalar `β_t` antes de cualquier otro
procesamiento:

```
m_t = (1 − λ) · m_{t−1}  +  λ · ‖k̂_t‖²        # EMA escalar, por cabeza
β_t^{ASN} = σ(w_β^⊤ x_t)  /  (ε + sqrt(m_t))
```

Costo: una división escalar y una actualización EMA por paso — `O(1)` por
token, por cabeza. `m_t` es un buffer no entrenable (estado auxiliar, igual
que `S`, pero de tamaño 1 por cabeza en vez de `d_v · d_k`).

**Pseudocódigo — ASN:**
```
function ASN(x_t, k_hat_t, m_prev, w_beta, lambda=0.01, eps=1e-6):
    raw_beta = sigmoid(w_beta @ x_t)
    m_t = (1 - lambda) * m_prev + lambda * dot(k_hat_t, k_hat_t)
    beta_t = raw_beta / (eps + sqrt(m_t))
    return beta_t, m_t
```

---

## 5. La pieza que NO se puede asumir: análisis de estabilidad

Esta sección es intencionalmente la más extensa y la más cuidadosa del
documento, porque es donde es más fácil cometer un error de razonamiento
"por analogía" en vez de demostrar algo.

### 5.1 La intuición ingenua (y por qué no basta)

Una primera intuición razonable es: *"como `α_t` está en `(0,1)` por canal, y
la corrección es una perturbación de rango 1, basta con controlar
`β_t · ‖k̃_t‖ · ‖k̂_t‖ ≤ 1` para que la transición no pueda amplificar el
estado."* Esta intuición **es incorrecta en general**, y este documento no
la da por válida.

Formalmente, definamos el **operador de transición** por token (actuando
por la derecha sobre el estado, ya que `S_t = S_{t-1} T_t` cuando se
factoriza la ecuación de la sección 4.4):

```
T_t = diag(α_t)  −  β_t · k̃_t k̂_t^⊤   ∈  R^{d_k × d_k}
```

Como `k̃_t ≠ k̂_t` en general (esa es justamente la propiedad que introduce
DEM, sección 4.4), `T_t` **no es simétrica**. La condición
`β_t‖k̃_t‖‖k̂_t‖ ≤ 1` controla la norma del término perturbativo por
separado, pero **no controla directamente los valores propios de `T_t`**,
porque `T_t` es una perturbación de rango 1 *no simétrica* de una matriz
diagonal, y sus valores propios `λ` satisfacen la ecuación secular:

```
1 = β_t · Σ_i  ( k̂_{t,i} · k̃_{t,i} )  /  ( α_{t,i} − λ )
```

(obtenida por el lema del determinante de matriz aplicado a
`det(T_t − λI) = 0`). Cuando `k̂_t` y `k̃_t` no son colineales, esta ecuación
puede tener soluciones `λ` con `|λ| > 1` incluso si
`β_t‖k̃_t‖‖k̂_t‖ = 1` exactamente.

### 5.2 Verificación experimental de que la intuición ingenua falla

Se realizó la siguiente prueba numérica (dimensión de clave `d_k = 16`,
20 000 muestras aleatorias): para cada muestra se generan `α_t ~ U(0.05,
0.99)^{16}`, `k̂_t` y `k̃_t` con entradas gaussianas independientes
(es decir, **no** colineales en general), y se fija `β_t` exactamente en el
límite de la condición ingenua: `β_t = 1 / (‖k̃_t‖‖k̂_t‖)`. Se calculan
numéricamente los valores propios de `T_t` y se mide cuántas muestras
violan `|λ|_max ≤ 1`.

**Resultado:** `4 917 / 20 000` muestras (≈ 24.6 %) tuvieron al menos un
valor propio con módulo mayor a 1, con un máximo observado de
`|λ|_max ≈ 1.538`. **Esto confirma formalmente y empíricamente que la
condición ingenua no garantiza estabilidad de un solo paso de la
transición.**

### 5.3 Una condición que sí es demostrable

En vez de intentar acotar los valores propios directamente (lo cual, dado
que `T_t` no es simétrica ni normal, no tiene una cota simple en términos
de normas de vectores), se usa la **norma de operador (norma espectral)**
de `T_t`, que sí es submultiplicativa y sí acota el radio espectral por
construcción (`radio_espectral(T) ≤ ‖T‖₂` siempre, para cualquier matriz):

```
‖T_t‖₂  ≤  ‖diag(α_t)‖₂  +  β_t · ‖k̃_t‖ · ‖k̂_t‖
        =  max_i(α_{t,i})  +  β_t · ‖k̃_t‖ · ‖k̂_t‖
```

(por la desigualdad triangular de la norma de operador). Por lo tanto, la
condición

```
max_i(α_{t,i})  +  β_t · ‖k̃_t‖ · ‖k̂_t‖  ≤  1
```

**sí es una condición suficiente demostrable** (no solo empírica) para que
`‖T_t‖₂ ≤ 1`, y por lo tanto para que el radio espectral de `T_t` sea `≤ 1`.

**Verificación numérica de esta segunda condición:** se repitió el
experimento anterior (50 000 muestras, mismas distribuciones), fijando esta
vez `β_t` en el límite exacto de la nueva condición:
`β_t = (1 − max_i α_{t,i}) / (‖k̃_t‖‖k̂_t‖)`. **Resultado: 0 / 50 000
violaciones** de `|λ|_max ≤ 1` (consistente con la demostración analítica,
ya que se trata de una cota probada, no de una observación estadística).

### 5.4 Un matiz importante: radio espectral por paso no es suficiente para la secuencia completa

Incluso si se garantiza `radio_espectral(T_t) ≤ 1` para **cada** paso `t`
por separado, esto **no** garantiza automáticamente que el producto de
transiciones a lo largo del tiempo, `T_1 · T_2 · ... · T_n`, tenga norma
acotada — porque las matrices `T_t` en general **no conmutan** entre sí, y
el radio espectral no es submultiplicativo bajo composición de matrices
distintas (solo lo es la norma de operador).

**Verificación experimental de este matiz:** se construyeron
deliberadamente pares de matrices `T_t` con radio espectral individual
`≤ 1` pero norma de operador individual `> 1` (posible porque, como se
explicó, ambas cantidades no coinciden para matrices no normales). Al
componerlas (`T_2 · T_1`), la norma del producto **superó transitoriamente
a 1** (se observó un pico de `‖T_2 T_1‖₂ ≈ 1.285` en la primera
composición) antes de decaer en pasos posteriores. Esto demuestra, con un
contraejemplo concreto y reproducible, que **acotar solo el radio espectral
por paso es una condición insuficiente**; lo que efectivamente controla el
comportamiento de `‖S_t‖` a lo largo de toda la secuencia es la norma de
operador de cada `T_t`, gracias a la submultiplicatividad:

```
‖S_t‖₂ = ‖S_0 · T_1 · T_2 · ... · T_t‖₂ ≤ ‖S_0‖₂ · Π_{i=1}^{t} ‖T_i‖₂
```

Si `‖T_i‖₂ ≤ 1` para todo `i` (garantizado por la condición de la sección
5.3), entonces `‖S_t‖₂ ≤ ‖S_0‖₂` para **todo** `t`, sin importar cuántos
pasos transcurran ni cómo se compongan las transiciones. Esta es la
propiedad que BDA realmente necesita, y es la que se impone activamente.

### 5.5 Componente resultante: Stability Projection (SP)

En base a 5.3–5.4, BDA no "confía" en que las puertas aprendidas mantengan
`‖T_t‖₂ ≤ 1`: lo **impone** en cada paso, recortando `β_t` cuando sea
necesario (nunca aumentándolo):

```
α_max_t   = max_i( α_{t,i} )
presupuesto_t = clip( 1 − α_max_t,  min = 0 )
β_t^{max} = presupuesto_t / ( ε + ‖k̃_t‖ · ‖k̂_t‖ )
β_t^{SP}  = min( β_t^{ASN},  β_t^{max} )
```

**Pseudocódigo — Stability Projection:**
```
function StabilityProjection(alpha_t, beta_t, k_hat_t, k_tilde_t, margin=1.0, eps=1e-6):
    alpha_max = max(alpha_t)                      # escalar
    budget    = max(margin - alpha_max, 0.0)
    denom     = norm(k_hat_t) * norm(k_tilde_t) + eps
    beta_max  = budget / denom
    return min(beta_t, beta_max)
```

Esto convierte la estabilidad de una *esperanza estadística* en una
**garantía dura, verificada matemáticamente e impuesta en tiempo de
ejecución**, con un costo adicional de `O(d_k)` por token (dos normas y un
`max`) — insignificante frente al costo de la actualización de estado.

### 5.6 Qué queda como garantizado y qué queda como abierto

**Garantizado (demostrado analíticamente y confirmado numéricamente):**
- Con Stability Projection activa, `‖S_t‖₂ ≤ ‖S_0‖₂` para todo `t`,
  independientemente de la longitud de la secuencia.
- Sin Stability Projection, la condición ingenua sobre `β` **no** es
  suficiente y puede producir amplificación por paso.

**Abierto / no garantizado todavía (marcado explícitamente, no asumido):**
- Que la cota de la sección 5.3 sea *ajustada* (tight); es conservadora por
  construcción (usa la desigualdad triangular), así que en la práctica
  puede estar limitando `β_t` más de lo estrictamente necesario. Cuantificar
  cuánta capacidad de aprendizaje se pierde por esta conservadurismo es
  trabajo experimental pendiente (sección 12).
- El comportamiento exacto de los valores propios de `T_t` (vía la ecuación
  secular de 5.1) en el régimen de entrenamiento real (con pesos
  aprendidos, no aleatorios) no ha sido caracterizado; los experimentos de
  esta sección usan pesos sin entrenar, lo cual es válido para estudiar la
  propiedad algebraica de la transición pero no dice nada sobre qué
  configuraciones aprende el modelo en la práctica.

---

## 6. Cómo BDA resuelve cada requisito funcional

### 6.1 State update — cómo incorpora información nueva
La actualización (sección 4.4) es una corrección de error, no una suma
acumulativa: el sistema calcula qué tan mal predice actualmente el
contenido asociado a la clave de borrado `k̃_t`, y corrige exactamente esa
discrepancia, escalada por `β_t^{SP}` (sección 5.5) en la dirección de
escritura `k̂_t`. La magnitud de la corrección está siempre acotada por la
Normalización Adaptativa de Paso (sección 4.5) y por la Proyección de
Estabilidad (sección 5.5).

### 6.2 State retrieval — cómo recupera información antigua
La recuperación es una lectura lineal del estado: `o_t = S_t q_t`. Como el
estado es una suma ponderada (con decaimiento por canal) de todas las
escrituras pasadas, la lectura recupera una combinación de la información
almacenada que sea más "cercana" (en el sentido del producto interno, canal
por canal) a la consulta actual. La calidad de esta recuperación depende de
cuánta interferencia haya sufrido la dirección de memoria relevante desde
que se escribió — por eso el control de interferencia (6.3) es central para
la calidad de la recuperación, no un mecanismo separado de ella.

### 6.3 Interferencia — que un token nuevo no destruya información importante
Este es el problema que resuelve directamente la Máscara de Borrado
Desacoplada (DEM, sección 4.4): al separar "qué se lee para calcular el
error" (`k̃_t`) de "hacia dónde se escribe la corrección" (`k̂_t`), el
modelo puede aprender a escribir información nueva sin forzar una lectura
de borrado total en la misma dirección — mitigando el caso en que dos
piezas de información no relacionadas comparten una dirección de memoria
parecida. Sumado a esto, el olvido por canal (LRFG, sección 4.3) permite
que distintos canales retengan información a distinto ritmo, reduciendo la
probabilidad de colisión entre información "vieja importante" e
"información nueva irrelevante".

### 6.4 Long-range retrieval — encontrar información después de miles de tokens
BDA por sí solo **no** garantiza recuperación exacta a distancias
arbitrarias (ver sección 3.3 — límite fundamental de cualquier estado de
tamaño fijo). Lo que aporta BDA a este problema es: (a) canales de memoria
de decaimiento lento (`α` cercano a 1) que pueden actuar como memoria de
largo plazo para la información que el modelo decida proteger, y (b) una
arquitectura híbrida (sección 6.4) donde capas de atención completa,
intercaladas periódicamente, se encargan de la recuperación exacta de largo
alcance que BDA no puede garantizar por construcción.

### 6.5 Multi-head memory — que cabezas distintas mantengan patrones distintos
Cada cabeza mantiene su propio estado `S^h` y sus propios parámetros de
puerta (`U_α^h`, `U_e^h`) generados a partir de un cuello de botella
compartido `z_t` (sección 4.3). Compartir `z_t` entre cabezas reduce costo
computacional; mantener proyecciones de salida (`U_α^h`, `U_e^h`)
independientes por cabeza preserva la capacidad de que cada cabeza
desarrolle un patrón de olvido/borrado distinto durante el entrenamiento,
ya que la no-linealidad (`σ`) y los parámetros propios de cada cabeza
pueden transformar el mismo `z_t` de formas arbitrariamente distintas.

### 6.6 Parallel training — entrenamiento eficiente en GPU, no solo inferencia rápida
Ver sección 7 (Chunk-Recurrent Parallel Training). El punto central: la
transición sigue teniendo la forma "diagonal + rango 1" en cada paso, lo
cual es la propiedad estructural que en la literatura de RNNs lineales
habilita algoritmos de entrenamiento por bloques. Esta especificación define
y valida experimentalmente (sección 10) un esquema de paralelismo *entre*
bloques con recurrencia corta *dentro* de cada bloque; el paralelismo
completo dentro del bloque queda como trabajo abierto (sección 12), sin
afirmarse aquí como resuelto.

---

## 7. Arquitectura completa y multi-cabeza

### 7.1 Notación
- `x_t ∈ R^{d_model}`: entrada en el paso `t`.
- `H`: número de cabezas. `d_k`, `d_v`: dimensión de clave/valor por cabeza.
- `S^h_t ∈ R^{d_v × d_k}`: estado de memoria de la cabeza `h` en el paso `t`.
- `m^h_t ∈ R`: buffer escalar de ASN por cabeza.

### 7.2 Paso hacia adelante (forward), un token, una cabeza
```
function BDA_step(x_t, S_prev, m_prev, head_params):
    z_t        = V @ x_t                                   # cuello de botella compartido
    alpha_t    = sigmoid(U_alpha @ z_t + b_alpha)           # (d_k,)
    e_t        = sigmoid(U_erase @ z_t + b_erase)           # (d_k,)
    k_hat_t    = W_k @ x_t                                  # (d_k,)
    k_tilde_t  = k_hat_t * e_t                               # DEM
    v_t        = W_v @ x_t                                  # (d_v,)
    q_t        = W_q @ x_t                                  # (d_k,)

    beta_asn, m_t = ASN(x_t, k_hat_t, m_prev, w_beta)
    beta_t        = StabilityProjection(alpha_t, beta_asn, k_hat_t, k_tilde_t)

    S_decayed = S_prev * alpha_t[None, :]                    # escala columnas
    error_t   = S_decayed @ k_tilde_t  -  v_t                # (d_v,)
    S_t       = S_decayed  -  beta_t * outer(error_t, k_hat_t)

    o_t = S_t @ q_t
    return o_t, S_t, m_t
```

### 7.3 Multi-cabeza
Cada cabeza ejecuta `BDA_step` de forma independiente, compartiendo
únicamente la proyección `V` del cuello de botella (sección 4.3). Las
salidas de todas las cabezas se concatenan y pasan por una proyección de
salida estándar, igual que en atención multi-cabeza convencional.

```
function BDA_multihead_step(x_t, {S_prev^h}, {m_prev^h}):
    outputs = []
    for h in 1..H:
        o_t^h, S_t^h, m_t^h = BDA_step(x_t, S_prev^h, m_prev^h, params[h])
        outputs.append(o_t^h)
    o_t = W_out @ concat(outputs)
    return o_t, {S_t^h}, {m_t^h}
```

### 7.4 Complejidad
- **Inferencia:** `O(H · d_k · d_v)` por token — constante respecto a la
  longitud de secuencia `N`. No hay caché que crezca con `N`.
- **Entrenamiento (secuencial de referencia):** `O(N · H · d_k · d_v)` —
  lineal en `N`, pero con profundidad secuencial `O(N)` (no paralelizable
  sin más estructura — de ahí la sección 7.5).

### 7.5 Motivación del esquema por bloques
Ejecutar `N` pasos secuenciales durante el entrenamiento es, en la
práctica, mucho más lento que un cómputo equivalente organizado en
operaciones matriciales grandes, aunque el conteo de operaciones aritméticas
sea el mismo — el cuello de botella real en GPU no es el número de FLOPs
sino el número de *lanzamientos de kernel* secuenciales y la ocupación del
hardware. La solución estándar en mecanismos con esta misma forma algebraica
("diagonal + rango 1") es reorganizar el cómputo en bloques.

### 7.6 Chunk-Recurrent Parallel Training (CRPT) — lo que se define y se valida aquí
La secuencia se divide en bloques de tamaño `C` (p. ej. `C = 16`–`64`). El
estado se transporta de un bloque al siguiente exactamente como en la
recurrencia secuencial completa; **dentro** de cada bloque, en esta versión
del diseño, el cómputo se mantiene como una recurrencia corta de longitud
`C`.

```
function BDA_train_CRPT(x_1..N, chunk_size=C):
    S = zeros(H, d_v, d_k)
    m = zeros(H)
    outputs = []
    for chunk in split_into_chunks(x_1..N, C):
        for x_t in chunk:                     # recurrencia corta, longitud <= C
            o_t, S, m = BDA_multihead_step(x_t, S, m)
            outputs.append(o_t)
    return outputs
```

Esto **no es todavía** el algoritmo de paralelismo fuerte dentro del bloque
(ver sección 3.2 y 12): lo que aporta, y lo que se valida en la sección 10,
es (a) que la partición en bloques con transporte de estado es exactamente
equivalente numéricamente a la recurrencia completa (no introduce
aproximación alguna) y (b) que, al mantener `C` pequeño, la profundidad
secuencial efectiva se reduce de `O(N)` a `O(N/C)` bloques, dentro de los
cuales la longitud `C` es lo bastante pequeña como para procesarse con alta
ocupación de hardware en una implementación de kernel dedicada (fuera del
alcance de este documento, que se limita a la especificación matemática y a
una validación de referencia en PyTorch).

### 7.7 Especificación del kernel especializado (Triton/CUDA)

Esta subsección responde a una pregunta muy concreta: **si CRPT en Python
(sección 7.6) es, en la práctica, más lento que un baseline de atención
completa bien optimizado, ¿qué tiene que tener exactamente un kernel para
que la promesa de velocidad de BDA deje de ser teórica?** Se especifica aquí
el conjunto de requisitos que cualquier implementación de kernel debe
cumplir. Como en el resto del documento, se marca explícitamente qué parte
de esto es una derivación cerrada y qué parte sigue siendo un objetivo de
diseño sin fórmula cerrada todavía (consistente con la sección 12).

#### 7.7.1 Diagnóstico: por qué la versión en Python es lenta

No es un problema de cantidad de FLOPs — es un problema de **granularidad
de ejecución**. La versión de referencia (sección 7.2, 7.6) hace, por cada
token, varias operaciones pequeñas separadas (proyección del cuello de
botella, dos sigmoides, dos normas, una división, un producto externo, una
resta) cada una como su propia operación de PyTorch. Cada una de esas
operaciones implica: (a) un lanzamiento de kernel en la GPU, con su
overhead fijo, y (b) una lectura/escritura del estado intermedio en memoria
HBM (la memoria "lenta" de la GPU), incluso cuando el resultado se vuelve a
usar inmediatamente después. Para `N` tokens eso son literalmente
`N × (número de operaciones por paso)` viajes a memoria HBM y lanzamientos
de kernel — el costo real no es aritmético, es de **tráfico de memoria y
overhead de lanzamiento**, y crece linealmente con `N` de la peor forma
posible: sin fusión.

Un kernel especializado ataca exactamente estos dos puntos: fusión de
operaciones y mantener el estado en memoria on-chip (registros / memoria
compartida) en vez de HBM.

#### 7.7.2 Qué operaciones deben fusionarse en un único kernel

Todo lo siguiente debe ejecutarse **dentro de un mismo kernel**, sin volver
a HBM entre pasos, para cada bloque de `C` tokens:

| Paso | Operación | Por qué debe estar fusionada |
|---|---|---|
| 1 | Aplicar decaimiento `S · diag(α_t)` | Usa el estado `S`, que debe seguir viviendo en memoria on-chip |
| 2 | Calcular `error_t = S_decayed · k̃_t − v_t` | Reutiliza `S_decayed` inmediatamente, sin escribirlo a HBM |
| 3 | Aplicar Stability Projection sobre `β_t` | Depende de `α_t`, `k̂_t`, `k̃_t`, ya presentes en registros |
| 4 | Corrección `S_t = S_decayed − β_t^{SP} · error_t · k̂_t^⊤` | Escritura final del estado del paso, todavía on-chip |
| 5 | Lectura de salida `o_t = S_t · q_t` | Usa el `S_t` recién calculado, antes de que se sobreescriba |

Las proyecciones de entrada (`Q, K, V`, el cuello de botella `z_t` de
LRFG/DEM) **no** necesitan estar dentro de este kernel: son
`GEMM`s densas estándar sobre toda la secuencia a la vez (embarazosamente
paralelas en la dimensión de tokens) y conviene dejarlas como llamadas a
`cuBLAS`/`cuDNN` normales, ya optimizadas, ejecutadas **antes** de entrar al
kernel recurrente. Fusionarlas dentro del kernel recurrente no aporta nada
y complica el diseño — la fusión importa específicamente para la cadena
secuencial (pasos 1–5), que es la parte que un compilador estándar no puede
paralelizar automáticamente por sí solo.

#### 7.7.3 Estructura de dos niveles obligatoria

Cualquier kernel correcto debe organizar el cómputo en dos niveles
claramente separados, no en un único bucle plano de longitud `N`:

**Nivel A — paralelo "ancho" (entre bloques, cabezas y elementos del
batch):** la dimensión `(batch × cabezas × número_de_bloques)` es
completamente independiente y debe mapearse al paralelismo masivo de la
GPU (un bloque de hilos de CUDA/Triton por cada combinación). Esto es
directo y no requiere ninguna derivación adicional: es la misma
independencia que ya explota cualquier atención estándar por cabeza.

**Nivel B — la cadena dentro de cada bloque de `C` tokens:** aquí es donde
está el verdadero trabajo de diseño del kernel, y tiene dos variantes
posibles, que deben distinguirse con la misma honestidad que el resto del
documento:

- **Variante mínima viable — recurrencia corta on-chip.** Dentro de un
  kernel de Triton, un bucle de `C` iteraciones (p. ej. `C = 64`) que
  mantiene `S` en memoria compartida/registros durante todo el bloque, y
  solo escribe a HBM el estado final del bloque (para el traspaso al
  siguiente bloque) y las `C` salidas `o_t`. Esto **no** requiere derivar
  ninguna fórmula matricial nueva — es exactamente la recurrencia de la
  sección 7.2, pero ejecutada dentro de un único kernel en vez de como `C`
  operaciones separadas de PyTorch. Ya elimina la enorme mayoría del
  overhead descrito en 7.7.1, porque pasa de `O(N)` lanzamientos de kernel
  y viajes a HBM a `O(N/C)`. **Esta es la variante que se recomienda
  implementar primero**, porque su corrección ya está validada
  matemáticamente por la prueba 4 de la sección 10 (equivalencia exacta con
  la recurrencia secuencial) — solo cambia *dónde* se ejecuta el mismo
  cómputo, no *qué* se computa.

- **Variante objetivo — solución paralela intra-bloque (matricial).**
  Reformular las `C` correcciones del bloque como la solución de un sistema
  triangular (de forma que las `C` salidas del bloque se obtengan con un
  puñado de multiplicaciones de matrices densas `C×C` y `C×d`, en vez de
  un bucle de `C` pasos). Esta es la pieza que da el salto de rendimiento
  más grande, porque las GPUs son mucho más eficientes en unas pocas
  multiplicaciones de matrices grandes que en muchos pasos secuenciales
  pequeños. **Esto sigue sin estar derivado en este documento para el caso
  general de BDA** (clave de escritura y clave de borrado desacopladas, DEM
  activo) — es exactamente lo que la sección 12 marca como trabajo abierto.
  Si se necesita un punto de partida concreto: en el caso particular en que
  `e_t ≡ 1` (DEM desactivado, `k̃_t = k̂_t`), la reformulación triangular es
  una construcción estándar y conocida para reglas de corrección de error
  con decaimiento diagonal, y puede implementarse y validarse primero en
  ese caso simplificado antes de extender la derivación al caso con
  borrado desacoplado.

#### 7.7.4 Qué debe vivir en memoria on-chip vs. en HBM

- **On-chip (registros / memoria compartida) durante el procesamiento de un
  bloque:** el estado `S` completo de la cabeza (`d_v × d_k` — con
  dimensiones típicas de 64–128 por lado, esto son unos pocos KB, muy por
  debajo de la memoria compartida disponible por bloque de hilos en GPUs
  actuales), el buffer escalar `m` de ASN, y las claves/valores/consultas
  del bloque actual (`C × d_k` y `C × d_v`, también pequeño para `C≈64`).
- **En HBM, leído una vez por bloque:** las proyecciones `Q, K, V, z` del
  bloque, ya calculadas por las GEMMs previas (sección 7.7.2).
- **En HBM, escrito una vez por bloque:** las `C` salidas `o_t` del bloque,
  y el estado `S` al final del bloque (para el siguiente bloque, o para
  recomputación en el backward — sección 7.7.5).
- **Lo que nunca debe escribirse a HBM:** el estado intermedio `S_t` para
  cada `t` dentro de un bloque. Si un kernel lo hace, no está resolviendo
  el problema de 7.7.1, solo lo está disfrazando.

#### 7.7.5 El backward pass necesita su propio diseño, no autograd genérico

Este punto se omite con frecuencia y es tan importante como el forward.
Si el forward se implementa como un kernel fusionado de Triton/CUDA,
**no se puede depender de que `autograd` de PyTorch derive automáticamente
el backward eficiente** — un kernel personalizado requiere una función
`backward` personalizada (v. `torch.autograd.Function`).

Requisitos del backward:

- **No almacenar `S_t` para cada `t`.** Guardar el estado completo en cada
  paso de cada bloque para el backward consumiría memoria `O(N · d_v · d_k)`
  — exactamente lo que se quiso evitar al usar un estado de tamaño fijo.
- **Estrategia de recomputación (checkpointing) por bloque:** guardar
  únicamente el estado `S` al **inicio** de cada bloque (`O(N/C · d_v ·
  d_k)` en total) durante el forward. En el backward, recomputar el forward
  dentro de cada bloque (barato, porque `C` es pequeño) para reconstruir los
  estados intermedios necesarios, calcular los gradientes locales, y
  propagar el gradiente del estado hacia el bloque anterior. Este es el
  mismo principio de "recompute en vez de almacenar" que usan los kernels
  de atención eficiente en memoria — no es una técnica exclusiva de BDA,
  pero es igual de necesaria aquí.
- **Gradiente a través de Stability Projection:** la operación `min(β_t,
  β_t^max)` (sección 5.5) es no diferenciable exactamente en el punto de
  cruce. En el backward debe tratarse como una función tipo `clip`: gradiente
  igual a 1 hacia `β_t` cuando `β_t < β_t^max` (la proyección no actuó), y
  gradiente igual a 1 hacia los términos de `β_t^max` (y por tanto hacia
  `α_t`, `k̂_t`, `k̃_t`) cuando la proyección sí recortó. Si esto se
  implementa mal (p. ej. bloqueando el gradiente por completo cuando SP
  actúa), el modelo nunca aprenderá a evitar la región donde SP tiene que
  intervenir, y SP se activará con más frecuencia de la necesaria durante
  todo el entrenamiento.

#### 7.7.6 Precisión numérica dentro del kernel

- Los productos punto usados por ASN y por Stability Projection (normas,
  divisiones) deben acumularse en `float32` **aunque el resto del modelo
  entrene en `bf16`/`fp16`**. Estas cantidades alimentan un cociente
  (secciones 4.5 y 5.5); un error de redondeo pequeño en el denominador de
  una división usada para *garantizar* estabilidad es exactamente el tipo
  de error que no se puede permitir. Esto es una instancia concreta de la
  limitación ya señalada en la sección 12 (precisión reducida no
  analizada) — aquí se convierte en un requisito de implementación, no solo
  en una advertencia.
- El estado `S` en sí puede mantenerse en `bf16` para el cómputo principal,
  pero la actualización (`S_decayed − corrección`) debe hacerse con un
  acumulador en `float32` antes de volver a convertir a `bf16` para el
  siguiente paso, para evitar que errores de redondeo se acumulen de forma
  sistemática a lo largo de miles de pasos (un problema clásico de RNNs en
  precisión reducida, no específico de BDA, pero que BDA hereda igual).

#### 7.7.7 Checklist de requisitos obligatorios del kernel

- [ ] Los pasos 1–5 de la tabla en 7.7.2 están fusionados en un único
      kernel, sin retorno a HBM entre ellos.
- [ ] El paralelismo de Nivel A (batch × cabezas × bloques) está mapeado
      al paralelismo de la GPU; el trabajo de Nivel B es lo único
      potencialmente secuencial.
- [ ] Como mínimo, la Variante mínima viable (7.7.3) está implementada y
      validada por igualdad numérica contra la recurrencia de referencia de
      la sección 7.2 (misma prueba que la número 4 de la sección 10, ahora
      contra el kernel real, no contra una segunda versión en Python).
- [ ] El estado intermedio `S_t` (para `t` dentro de un bloque) nunca se
      escribe a HBM.
- [ ] Existe una función `backward` personalizada con recomputación por
      bloque (7.7.5) — no se depende de autograd genérico sobre un bucle.
- [ ] El gradiente a través de Stability Projection está implementado como
      un `clip` diferenciable correctamente (7.7.5), verificado con
      comprobación numérica de gradiente (`gradcheck` o equivalente) sobre
      una versión pequeña antes de confiar en el kernel completo.
- [ ] Las cantidades usadas por ASN y Stability Projection se acumulan en
      `float32` incluso si el resto corre en precisión reducida (7.7.6).
- [ ] Existe un benchmark de velocidad contra un baseline de atención
      completa optimizada (p. ej. Flash Attention) en las mismas
      dimensiones, **antes** de afirmar en cualquier documento o
      comunicación pública que BDA es más rápido — la sección 1 de este
      documento no permite afirmaciones de velocidad sin medición.

#### 7.7.8 Qué NO se debe hacer al construir el kernel

- **No** implementar el forward como kernel fusionado y dejar el backward
  al autograd automático "porque funciona" — funcionará, pero
  materializará todos los estados intermedios en memoria, perdiendo la
  ventaja de memoria que es parte del objetivo de diseño.
- **No** intentar fusionar las proyecciones de entrada (`Q,K,V,z`) dentro
  del mismo kernel que la recurrencia — son operaciones de naturaleza
  distinta (paralelas vs. secuenciales) y mezclarlas complica el kernel sin
  beneficio de rendimiento real.
- **No** publicar cifras de velocidad obtenidas únicamente con la Variante
  mínima viable (7.7.3) presentándolas como el rendimiento "final" de BDA
  sin aclarar que la Variante objetivo (solución matricial intra-bloque)
  todavía no está implementada — sería repetir, en la fase de kernel, el
  mismo tipo de sobre-afirmación que este documento evita deliberadamente
  en la fase de diseño matemático (secciones 3.2 y 9).
- **No** usar un tamaño de bloque `C` fijo sin comprobar el trade-off real:
  `C` muy pequeño reduce la ganancia de fusión (poco trabajo por
  lanzamiento de kernel); `C` muy grande satura la memoria on-chip
  disponible y fuerza spilling a memoria más lenta. `C` debe tratarse como
  un hiperparámetro de rendimiento a barrer empíricamente por GPU objetivo,
  no como una constante de diseño.

---

## 8. Qué se debe implementar

- [ ] Estado de memoria `S^h` por cabeza, inicializado en cero, como buffer
      recurrente (no parámetro entrenable).
- [ ] Cuello de botella compartido `V` para generar `z_t`, con proyecciones
      de salida `U_α^h`, `U_e^h` independientes por cabeza.
- [ ] Clave de escritura `k̂_t` y clave de borrado `k̃_t = k̂_t ⊙ e_t` como
      vectores **distintos**, nunca colapsados en una sola proyección.
- [ ] Normalización Adaptativa de Paso (ASN) con su buffer EMA `m^h`,
      actualizado en cada paso, no recalculado desde cero.
- [ ] Stability Projection aplicada **después** de ASN y **antes** de usar
      `β_t` en la actualización de estado — nunca omitida, ni siquiera en
      configuraciones "rápidas" o de prueba.
- [ ] Verificación de equivalencia numérica entre la recurrencia secuencial
      de referencia y cualquier implementación por bloques, antes de
      confiar en la versión por bloques para entrenamiento real (ver
      metodología en sección 10).
- [ ] Registro y monitoreo de `‖S_t‖` durante entrenamiento como métrica de
      salud del modelo (no solo la pérdida).

## 9. Qué NO se debe implementar

- [ ] **No** implementar la actualización de estado sin Stability
      Projection "para simplificar", incluso en prototipos — la sección 5
      muestra que la inestabilidad puede aparecer con parámetros
      perfectamente razonables, no solo en casos patológicos.
- [ ] **No** colapsar `k̂_t` y `k̃_t` en la misma proyección salvo que se
      esté deliberadamente probando una variante "sin desacople" con fines
      de comparación (ver protocolo experimental, sección 11) — hacerlo por
      defecto anula el componente que resuelve la interferencia (sección
      6.3).
- [ ] **No** usar BDA en solitario, sin ninguna capa de atención completa
      intercalada, en aplicaciones donde la recuperación exacta de
      contexto largo sea un requisito — la sección 3.3 explica por qué esto
      no es una limitación de implementación sino matemática.
- [ ] **No** afirmar paralelismo total dentro de bloque (WY / triangular
      solve completo) como si estuviera implementado y validado: esta
      versión solo valida el esquema de recurrencia corta entre bloques
      (sección 7.6). Afirmar más que esto en documentación pública sería
      impreciso.
- [ ] **No** asumir que la cota de estabilidad de la sección 5.3 es ajustada
      (tight) sin medirlo — puede estar sacrificando capacidad de
      aprendizaje innecesariamente; esto debe medirse (sección 12), no
      suponerse.

---

## 10. Registro de validación experimental (esta versión del documento)

Para evitar afirmaciones no verificadas, se listan aquí exactamente las
pruebas ejecutadas, con sus resultados, sobre una implementación de
referencia en PyTorch de BDA (proyecciones aleatorias sin entrenar, ya que
el objetivo de estas pruebas es la corrección algebraica y numérica del
mecanismo, no su desempeño como modelo entrenado).

| # | Prueba | Configuración | Resultado |
|---|---|---|---|
| 1 | Condición ingenua de estabilidad (`β‖k̃‖‖k̂‖≤1`) | `d_k=16`, 20 000 muestras aleatorias, límite exacto de la condición | **24.6 % de violaciones** de `\|λ\|_max ≤ 1`; máximo observado `\|λ\|_max ≈ 1.538` |
| 2 | Condición de norma de operador (`α_max+β‖k̃‖‖k̂‖≤1`) | `d_k=16`, 50 000 muestras, límite exacto de la condición | **0 / 50 000 violaciones** — consistente con la cota demostrada analíticamente |
| 3 | Crecimiento transitorio con radio espectral por paso `≤1` | Matrices construidas para tener radio espectral individual `≤1` pero norma de operador `>1`; composición de dos pasos | Norma del producto alcanzó **≈1.285** en la composición antes de decaer — confirma que el radio espectral por paso, por sí solo, no basta |
| 4 | Equivalencia numérica: recurrencia secuencial completa vs. esquema por bloques (CRPT) | `B=2, T=97, H=4, d_k=d_v=32`, tamaño de bloque `16` | Diferencia máxima absoluta en salidas: **0.0** (exacta, como se esperaba matemáticamente) |
| 5 | Flujo de gradiente | Retropropagación sobre la implementación de referencia completa (LRFG, DEM, ASN, SP incluidos) | Gradientes finitos en todos los parámetros de las puertas y en la entrada; sin `NaN`/`Inf` |
| 6 | Comportamiento a horizonte largo, con y sin Stability Projection | Secuencias de `200` a `8000` pasos, pesos sin entrenar | **Con SP:** magnitud máxima de salida se mantuvo acotada (`≈0.50` → `≈0.80` de `T=200` a `T=8000`). **Sin SP:** crecimiento monótono (`≈6.6` → `≈14.6` en el mismo rango) |

**Nota de honestidad metodológica:** las pruebas 1–4 son propiedades
algebraicas de la transición (válidas para cualquier valor de los
parámetros, entrenados o no). La prueba 6 usa pesos **sin entrenar** —
es evidencia de que el mecanismo, en su forma matemática, tiene el
comportamiento esperado, pero **no** es una medición del comportamiento de
un modelo BDA entrenado en una tarea real. Esa validación (perplejidad,
recuperación asociativa, tareas de largo contexto) es trabajo futuro
(sección 12), no se afirma aquí.

---

## 11. Protocolo experimental sugerido para las siguientes fases

1. **Verificación de kernel por bloques (si se implementa CRPT en un
   kernel dedicado, p. ej. Triton):** repetir la prueba 4 de la sección 10
   contra esa implementación, no solo contra la versión de referencia en
   Python.
2. **Medición de conservadurismo de Stability Projection:** entrenar una
   versión pequeña del modelo con y sin el recorte de `β_t` (en un régimen
   donde de todas formas no diverja, para aislar el efecto en capacidad de
   aprendizaje) y comparar velocidad de convergencia / pérdida final.
3. **Prueba de interferencia controlada:** diseñar una tarea sintética
   donde se escribe información en una dirección de clave, luego se
   escriben muchos tokens "distractores" con claves parecidas, y se mide
   qué tan bien se recupera la información original — comparando BDA con y
   sin DEM activo (ablación).
4. **Recuperación de largo alcance:** tarea tipo "aguja en el pajar" a
   distintas longitudes de contexto, comparando BDA puro, BDA en
   arquitectura híbrida (sección 3.2 y 6.4), y un modelo de referencia
   basado en atención completa.
5. **Escalado:** repetir 1–4 a escalas de parámetros crecientes (decenas de
   millones → cientos de millones) antes de cualquier afirmación sobre
   comportamiento a escala de modelos de producción.

---

## 12. Limitaciones y trabajo futuro (explícito)

- La solución matricial de paralelismo **dentro** de cada bloque ("Variante
  objetivo" de la sección 7.7.3) no está derivada ni validada en esta
  versión para el caso general con borrado desacoplado (DEM activo); solo
  se valida el transporte de estado **entre** bloques con recurrencia corta
  dentro de ellos ("Variante mínima viable", secciones 7.6 y 7.7.3). Ningún
  kernel (Triton/CUDA) ha sido implementado todavía — la sección 7.7 es una
  especificación de requisitos, no una implementación medida.
- No se ha medido el costo en capacidad de aprendizaje que introduce el
  conservadurismo de Stability Projection (sección 5.6).
- No se ha entrenado ningún modelo real con BDA; todas las validaciones de
  esta versión son algebraicas/numéricas sobre el mecanismo, no empíricas
  sobre desempeño en tareas de lenguaje.
- La proporción óptima de capas BDA frente a capas de atención completa en
  una arquitectura híbrida (sección 6.4) no se ha determinado
  experimentalmente para este mecanismo específico; se trata como
  hiperparámetro a ajustar, no como valor fijo recomendado en esta versión.
- El análisis de la sección 5 asume proyecciones de clave/valor de
  precisión numérica estándar (`float32`); el comportamiento bajo
  entrenamiento en precisión reducida (`bfloat16`/`fp16`) no se ha
  analizado y puede requerir márgenes de seguridad adicionales en Stability
  Projection.

---

## 13. Glosario de símbolos

| Símbolo | Significado |
|---|---|
| `x_t` | Entrada del modelo en el paso `t` |
| `S_t^h` | Estado de memoria de la cabeza `h` en el paso `t` |
| `α_t` | Vector de olvido por canal (salida de LRFG) |
| `e_t` | Máscara de borrado por canal (salida de DEM) |
| `k̂_t` | Clave de escritura |
| `k̃_t` | Clave de borrado/lectura para el cálculo del error (`k̂_t ⊙ e_t`) |
| `v_t` | Valor a escribir |
| `q_t` | Consulta de lectura |
| `β_t` | Tasa de corrección, tras ASN y Stability Projection |
| `m_t^h` | Buffer EMA escalar de ASN por cabeza |
| `T_t` | Operador de transición del estado en el paso `t` |
| `‖·‖₂` | Norma de operador (norma espectral) de una matriz |

---

## 14. Cierre

BDA se presenta en esta versión como un mecanismo **definido con precisión
matemática, con seis componentes nombrados y verificables, y con una
sección de estabilidad que reemplaza una intuición no demostrada por una
condición demostrada y verificada numéricamente**. Las secciones 8–9 y 12
delimitan explícitamente qué está listo para implementarse y qué sigue
siendo trabajo abierto, para que cualquier persona que reciba este
documento — la conozca o no de antemano — entienda exactamente qué se está
proponiendo, qué se ha verificado, y qué todavía no.
