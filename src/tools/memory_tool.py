import os
import json
import config

class MemoryManager:
    def __init__(self):
        self.memory_path = config.MEMORY_FILE_PATH
        self.memory = {
            "facts": {},
            "lessons_learned": [],
            "completed_tasks": []
        }
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                # Ensure all required keys exist
                if "facts" not in self.memory:
                    self.memory["facts"] = {}
                if "lessons_learned" not in self.memory:
                    self.memory["lessons_learned"] = []
                if "completed_tasks" not in self.memory:
                    self.memory["completed_tasks"] = []
            except Exception as e:
                print(f"Error loading memory: {e}")

    def save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def store_fact(self, key, value):
        self.memory["facts"][key] = value
        self.save_memory()
        return f"Stored fact: {key} = {value}"

    def append_lesson(self, lesson):
        if lesson not in self.memory["lessons_learned"]:
            self.memory["lessons_learned"].append(lesson)
            self.save_memory()
            return f"Added new lesson learned: '{lesson}'"
        return "Lesson already exists in memory."

    def log_completed_task(self, task_name, summary):
        self.memory["completed_tasks"].append({
            "task": task_name,
            "summary": summary
        })
        self.save_memory()
        return f"Logged completed task: {task_name}"

    def get_summary(self):
        return json.dumps(self.memory, indent=2)

# Global memory manager
memory_manager = MemoryManager()

# Tools
class RetrieveMemoryTool:
    name = "retrieve_memory"
    description = "Retrieve all stored long-term memories (facts, lessons learned, and completed tasks) to help solve the task."

    def __call__(self, query: str = "") -> str:
        memory_manager.load_memory() # Reload to get latest
        mem = memory_manager.memory
        if not query:
            return json.dumps(mem, indent=2)
        
        # If query is specified, do a simple string match filtering
        q = query.lower()
        filtered_facts = {k: v for k, v in mem["facts"].items() if q in k.lower() or q in str(v).lower()}
        filtered_lessons = [l for l in mem["lessons_learned"] if q in l.lower()]
        filtered_tasks = [t for t in mem["completed_tasks"] if q in t["task"].lower() or q in t["summary"].lower()]
        
        results = {
            "facts": filtered_facts,
            "lessons_learned": filtered_lessons,
            "completed_tasks": filtered_tasks
        }
        return json.dumps(results, indent=2)

class StoreMemoryTool:
    name = "store_memory"
    description = "Store a key-value fact in the long-term memory. Useful to persist facts across context resets."

    def __call__(self, key: str, value: str) -> str:
        return memory_manager.store_fact(key, value)

class StoreLessonTool:
    name = "store_lesson"
    description = "Store a self-improvement lesson learned. Call this when you discover a mistake or find a better way of doing something."

    def __call__(self, lesson: str) -> str:
        return memory_manager.append_lesson(lesson)
