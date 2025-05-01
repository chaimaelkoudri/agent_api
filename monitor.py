import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag.vector_store import create_vectorstore
import time
import shutil  # Per eliminare directory in modo ricorsivo

# Classe che osserva una cartella e reagisce ai cambiamenti
class Watcher:
    def __init__(self, folder_path, debounce_sec = 300):
        self.folder_path = folder_path  # Cartella da monitorare
        self.debounce_seconds = debounce_sec
        self.last_updated = 0

        # Definisce il gestore degli eventi (cosa succede quando i file cambiano)
        self.event_handler = FileSystemEventHandler()
        self.event_handler.on_any_event = self.on_modification  # Chiama questa funzione per ogni evento

        # Imposta l’osservatore che terrà d’occhio la cartella
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.folder_path, recursive=False)

    # Avvia l’osservazione
    def start(self):
        print(f"Monitoring changes in '{self.folder_path}'...")
        self.observer.start()

    # Ferma l’osservazione
    def stop(self):
        self.observer.stop()
        self.observer.join()

    # Funzione chiamata ogni volta che un file nella cartella viene modificato, creato o eliminato
    def on_modification(self, event):
        if event.is_directory:
            return  # Ignora le modifiche alle sottocartelle
        
        now = time.time()
        if now - self.last_updated < self.debounce_seconds:
            print(f"Change detected but waiting {int(self.debounce_seconds - (now - self.last_updated))}s before next update.")
            return

        print(f"Change detected: {event.src_path}")  # Stampa il file modificato
        self.update_vectorstore()  # Ricrea il vectorstore

    # Ricrea l’indice FAISS da zero
    def update_vectorstore(self):
        persist_path = "rag_index"  # Dove salviamo l'indice

        # Se l’indice esiste già, lo eliminiamo per ricostruirlo
        if os.path.exists(persist_path):
            print("Removing old vectorstore...")
            shutil.rmtree(persist_path)

        # Crea un nuovo indice con i documenti aggiornati
        print("Creating new vectorstore...")
        create_vectorstore(self.folder_path)
        print("Vectorstore updated!")

# Se esegui questo file direttamente (non importato da altri), esegui il monitor
if __name__ == "__main__":
    folder_path = "docs"  # Cartella contenente i PDF da monitorare
    watcher = Watcher(folder_path)

    try:
        watcher.start()  # Avvia l’osservazione
        while True:
            time.sleep(1)  # Mantiene lo script attivo
    except KeyboardInterrupt:
        watcher.stop()  # Permette di fermarlo con CTRL+C
        print("Monitoring stopped.")