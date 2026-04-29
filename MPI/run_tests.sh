#!/bin/bash

# Nome del file di output
CSV_FILE="mpi_benchmark_results.csv"

# Intestazione del file CSV
echo "N_Size,Num_Processes,Iterazioni,Tempo_Medio(s)" > $CSV_FILE

# Parametri dei test (ho ridotto i test grandi per farti fare prima una prova)
SIZES=(1024 2048 4096 8192)
PROCESSES=(1 2 4 8 16 24 32)
ITERATIONS=3

# === LA MODIFICA È QUI ===
# Usiamo il wrapper MPI di Intel. 
# Se il tuo sistema usa il nuovo compiler LLVM di Intel, usa CC=mpiicx
CC=mpiicc 
CFLAGS="-O3 -xHost -qopenmp-simd -diag-disable=10441"

echo "=== Inizio Benchmark MPI Matrix Multiplication (Intel Compiler) ==="

for N in "${SIZES[@]}"; do
    echo "[DEBUG] Compilazione in corso per N=$N, ITER=$ITERATIONS con $CC..."
    
    # Compilazione
    $CC $CFLAGS -DN=$N -DITER=$ITERATIONS matmul_mpi.c -o matmul_mpi_exec
    
    if [ $? -ne 0 ]; then
        echo "Errore di compilazione per N=$N. Assicurati che l'ambiente Intel sia caricato."
        exit 1
    fi

    for P in "${PROCESSES[@]}"; do
        echo "[DEBUG] Esecuzione con N=$N, Processi=$P..."
        
        OUTPUT=$(mpirun -np $P ./matmul_mpi_exec 2>/dev/null)
        RESULT_LINE=$(echo "$OUTPUT" | grep "^RESULT")
        
        if [ ! -z "$RESULT_LINE" ]; then
            TIME=$(echo "$RESULT_LINE" | awk -F',' '{print $4}')
            echo "$N,$P,$ITERATIONS,$TIME" >> $CSV_FILE
            echo "[DEBUG] Completato! Tempo medio: $TIME s"
        else
            echo "[DEBUG] Errore nell'esecuzione o out-of-memory per P=$P, N=$N"
            echo "$N,$P,$ITERATIONS,ERROR" >> $CSV_FILE
        fi
    done
    echo "---------------------------------------------------"
done

echo "=== Benchmark Concluso. Risultati salvati in $CSV_FILE ==="