# Analyse des erreurs TensorFlow Lite et leurs solutions

## 1. Erreurs avec les modèles à 8 sorties

### Erreur rencontrée
```
Error during inference: Cannot copy from a TensorFlowLite tensor (StatefulPartitionedCall:1) with shape [1, 1, 1] to a Java object with shape [1, 100, 100].
```

### Cause
Dans la méthode `runInferenceForEightOutputs`, nous avons créé des tableaux avec des dimensions fixes (par exemple `Array(1) { Array(100) { FloatArray(100) }}`) au lieu d'utiliser les dimensions réelles des tenseurs du modèle.

### Détails des tenseurs
```
Output 0: shape=[1, 12804, 4], dataType=FLOAT32, numBytes=204864
Output 1: shape=[1, 1], dataType=FLOAT32, numBytes=400
Output 2: shape=[1], dataType=FLOAT32, numBytes=4
Output 3: shape=[1, 12804, 2], dataType=FLOAT32, numBytes=102432
Output 4: shape=[1, 1, 1], dataType=FLOAT32, numBytes=1600
Output 5: shape=[1, 1], dataType=FLOAT32, numBytes=400
Output 6: shape=[1, 1], dataType=FLOAT32, numBytes=400
Output 7: shape=[1, 1, 1], dataType=FLOAT32, numBytes=800
```

### Solution
Utiliser les dimensions réelles des tenseurs pour créer les tableaux de sortie :

```kotlin
// Avant (incorrect)
outputs[4] = Array(1) { Array(100) { FloatArray(100) } }

// Après (correct)
val output4Shape = outputShapes[4]
val output4Array = Array(output4Shape[0]) { 
    Array(output4Shape[1]) { FloatArray(output4Shape[2]) } 
}
outputs[4] = output4Array
```

### Erreurs de manipulation des tableaux
Nous avons également rencontré des erreurs avec les propriétés `size` et `indices` :

```
Unresolved reference 'size'.
Method 'iterator()' is ambiguous for this expression.
Unresolved reference 'indices'.
```

#### Solution
Utiliser les propriétés appropriées selon le type de tableau :
- Pour `Array` : utiliser `.length` et `0 until array.length`
- Pour `FloatArray` : utiliser `.size` et `0 until array.size`

```kotlin
// Avant (incorrect)
val rawBoxes = FloatArray(rawBoxesArray[0].size * rawBoxesArray[0][0].size * rawBoxesArray[0][0][0].size)
for (i in rawBoxesArray[0].indices) {
    // ...
}

// Après (correct)
val rawBoxes = FloatArray(rawBoxesArray[0].size * 4)
for (i in 0 until rawBoxesArray[0].size) {
    // ...
}
```

## 2. Erreurs avec les modèles à 12 sorties

### Erreur rencontrée
```
Cannot convert between a TensorFlowLite tensor with type UINT8 and a Java object of type [[F (which is compatible with the TensorFlowLite type FLOAT32).
```

### Cause
Nous utilisions des `FloatArray` pour recevoir des données de tenseurs de type `UINT8`.

### Détails des tenseurs
```
Output 0: shape=[1, 20, 20, 12], dataType=UINT8, numBytes=4800
Output 1: shape=[1, 20, 20, 273], dataType=UINT8, numBytes=109200
Output 2: shape=[1, 1, 1, 24], dataType=UINT8, numBytes=24
...
```

### Solution
Utiliser des `ByteArray` au lieu de `FloatArray` pour les tenseurs de type `UINT8` :

```kotlin
// Avant (incorrect)
outputs[i] = FloatArray(shape[0])

// Après (correct)
outputs[i] = ByteArray(shape[0])
```

### Erreur de forme (shape)
```
Cannot copy from a TensorFlowLite tensor with shape [1, 20, 20, 12] to a Java object with shape [1, 20].
```

#### Cause
Notre code ne gérait pas correctement les tenseurs 4D.

#### Solution
Ajouter un support pour les tenseurs 4D :

```kotlin
// Avant (incorrect - ne gère pas les tenseurs 4D)
when (shape.size) {
    1 -> outputs[i] = ByteArray(shape[0])
    2 -> outputs[i] = Array(shape[0]) { ByteArray(shape[1]) }
    3 -> outputs[i] = Array(shape[0]) { Array(shape[1]) { ByteArray(shape[2]) } }
    else -> outputs[i] = Array(shape[0]) { ByteArray(shape[1]) }
}

// Après (correct - gère les tenseurs 4D)
when (shape.size) {
    1 -> outputs[i] = ByteArray(shape[0])
    2 -> outputs[i] = Array(shape[0]) { ByteArray(shape[1]) }
    3 -> outputs[i] = Array(shape[0]) { Array(shape[1]) { ByteArray(shape[2]) } }
    4 -> outputs[i] = Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { ByteArray(shape[3]) } } }
    else -> {
        Log.w(TAG, "Unsupported tensor shape size: ${shape.size} for output $i")
        outputs[i] = ByteArray(1)
    }
}
```

## Bonnes pratiques pour éviter ces erreurs

1. **Vérifier le type de données des tenseurs** avant de créer les tableaux de sortie
   ```kotlin
   val isUint8Model = interp.getOutputTensor(0).dataType() == org.tensorflow.lite.DataType.UINT8
   ```

2. **Inspecter les formes des tenseurs** et créer des structures de données correspondantes
   ```kotlin
   Log.d(TAG, "Output $i: shape=${shape.contentToString()}")
   ```

3. **Ajouter des logs détaillés** pour faciliter le débogage
   ```kotlin
   for (i in 0 until outputShapes.size) {
       val tensor = interp.getOutputTensor(i)
       Log.d(TAG, "Output $i: shape=${tensor.shape().contentToString()}, " +
             "dataType=${tensor.dataType()}, numBytes=${tensor.numBytes()}")
   }
   ```

4. **Gérer les cas d'erreur** avec des messages clairs et des fallbacks appropriés

5. **Adapter le traitement post-inférence** selon le type de données
   ```kotlin
   // Pour les modèles UINT8, convertir les bytes en floats
   fun byteToFloat(value: Byte): Float {
       return value.toInt().and(0xFF).toFloat()
   }
   ```

6. **Vérifier les dimensions avant d'accéder aux tableaux** pour éviter les IndexOutOfBoundsException
   ```kotlin
   if (i >= outputShapes.size) {
       Log.w(TAG, "Output index $i exceeds available shapes (${outputShapes.size})")
       continue
   }
   ```

Ces corrections permettent à votre code de s'adapter correctement aux différents formats de tenseurs, qu'ils soient de type FLOAT32 ou UINT8, et avec des dimensions variées (1D à 4D).

## Résumé des types de tableaux à utiliser

| Type de tenseur | Dimension | Type Java à utiliser |
|----------------|-----------|---------------------|
| FLOAT32        | 1D        | FloatArray          |
| FLOAT32        | 2D        | Array<FloatArray>   |
| FLOAT32        | 3D        | Array<Array<FloatArray>> |
| FLOAT32        | 4D        | Array<Array<Array<FloatArray>>> |
| UINT8          | 1D        | ByteArray           |
| UINT8          | 2D        | Array<ByteArray>    |
| UINT8          | 3D        | Array<Array<ByteArray>> |
| UINT8          | 4D        | Array<Array<Array<ByteArray>>> |
