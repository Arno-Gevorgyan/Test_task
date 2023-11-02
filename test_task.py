import heapq
import random
import logging
from threading import Lock
from dataclasses import dataclass
from typing import List, Optional

# Configure logging at the root level of the application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(order=True)
class Resources:
    """
    Represents the computational resources required by a task.

    Attributes:
        ram (int): Amount of RAM required.
        cpu_cores (int): Number of CPU cores required.
        gpu_count (int): Number of GPUs required.
    """
    ram: int
    cpu_cores: int
    gpu_count: int

    def is_sufficient(self, other) -> bool:
        """
        Determine if the available resources are sufficient to meet the required resources.

        Args:
            other (Resources): The available resources to compare against the required ones.

        Returns:
            bool: True if available resources are sufficient, False otherwise.
        """
        sufficient = all([
            other.ram >= self.ram,
            other.cpu_cores >= self.cpu_cores,
            other.gpu_count >= self.gpu_count
        ])
        if not sufficient:
            logging.warning(f"Insufficient resources: Required {self}, Available {other}")
        return sufficient


@dataclass(order=True)
class Task:
    """
    Represents a task with a given priority and resource requirements.

    Attributes:
        priority (int): Priority of the task, with higher numbers being higher priority.
        id (int): Unique identifier of the task.
        resources (Resources): Resources required to execute the task.
        content (str): The content or description of the task.
        result (str): The result of the task execution, empty by default.
    """
    priority: int
    id: int
    resources: Resources
    content: str
    result: str = ''


class Node:
    """A node in a doubly linked list that contains a value and pointers to the previous and next nodes."""

    def __init__(self, value: Task):
        """
        Initializes a new Node.

        Args:
            value (Task): The task value contained in the node.
        """
        self.value: Task = value
        self.prev: 'Optional[Node]' = None
        self.next: 'Optional[Node]' = None


class DoublyLinkedList:
    """A doubly linked list that allows for efficient insertion and removal of nodes."""

    def __init__(self):
        """
        Initializes a new empty DoublyLinkedList.
        """
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None

    def push_front(self, value: Task) -> Node:
        """
        Inserts a new node with the given value at the beginning of the list.

        Args:
            value (Task): The task value to insert at the front of the list.

        Returns:
            Node: The newly created node.
        """
        new_node = Node(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        return new_node

    def insert_after(self, node: Node, value: Task) -> Node:
        """
        Inserts a new node with the given value after the specified node.

        Args:
            node (Node): The node after which the new value should be inserted.
            value (Task): The task value to insert.

        Returns:
            Node: The newly created node.
        """
        if not node:
            return self.push_front(value)
        new_node = Node(value)
        new_node.prev = node
        new_node.next = node.next
        if node.next:
            node.next.prev = new_node
        else:
            self.tail = new_node
        node.next = new_node
        return new_node

    def remove(self, node: Node):
        """
        Removes the specified node from the list.

        Args:
            node (Node): The node to remove.
        """
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def front(self) -> Optional[Node]:
        """
        Gets the first node in the list.

        Returns:
            Optional[Node]: The first node in the list or None if the list is empty.
        """
        return self.head


class TaskQueue:
    """
    A thread-safe priority queue for tasks with resource constraints.

    Attributes:
        dll (DoublyLinkedList): A doubly linked list to store tasks.
        priority_map (Dict[int, Tuple[Node, Node]]): A mapping from task priorities to
                                                     tuples of the first and last nodes at
                                                     each priority level in the doubly linked list.
        lock (Lock): A threading lock to ensure thread-safe operations on the queue.
    """

    def __init__(self):
        """
        Initializes a new instance of TaskQueue.
        """
        self.dll = DoublyLinkedList()
        self.priority_map: Dict[int, Tuple[Node, Node]] = {}
        self.lock = Lock()

    def add_task(self, task: Task):
        """
        Adds a task to the queue in a thread-safe manner.

        Args:
            task (Task): The task to be added to the queue.

        Raises:
            ValueError: If the provided task is not an instance of Task.
        """
        if not isinstance(task, Task):
            raise ValueError("Only Task instances can be added to the queue.")

        with self.lock:
            logging.info(f"Adding task with ID {task.id} and priority {task.priority}")
            if task.priority not in self.priority_map:
                new_node = self.dll.push_front(task)
                self.priority_map[task.priority] = (new_node, new_node)
            else:
                _, last_node = self.priority_map[task.priority]
                new_node = self.dll.insert_after(last_node, task)
                self.priority_map[task.priority] = (self.priority_map[task.priority][0], new_node)

    def get_task(self, available_resources: Resources) -> Optional[Task]:
        """
        Retrieves the highest priority task that can be processed with the available resources.

        Args:
            available_resources (Resources): The resources available to process a task.

        Returns:
            Optional[Task]: The next task that can be processed given the available resources,
                            or None if no suitable task is available.
        """
        with self.lock:
            current = self.dll.front()
            while current:
                task = current.value
                if task.resources.is_sufficient(available_resources):
                    logging.info(f"Task with ID {task.id} fetched from the queue")
                    self.dll.remove(current)
                    self._update_priority_map_after_removal(task)
                    return task
                current = current.next

            logging.info("No suitable task found for the available resources.")
            return None

    def _update_priority_map_after_removal(self, task: Task):
        """
        Updates the priority map after a task has been removed from the queue.

        Args:
            task (Task): The task that has been removed.
        """
        first_node, last_node = self.priority_map[task.priority]
        if first_node == last_node:
            del self.priority_map[task.priority]
            logging.debug(f"Priority {task.priority} removed from the map after task removal")
        else:
            if first_node.value == task:
                self.priority_map[task.priority] = (first_node.next, last_node)
            elif last_node.value == task:
                self.priority_map[task.priority] = (first_node, last_node.prev)
            logging.debug(f"Priority map updated after task {task.id} removal")


def generate_random_task(task_id: int) -> Task:
    """
    Generates a random task with a unique ID and random resource requirements.

    Args:
        task_id (int): The unique identifier for the task.

    Returns:
        Task: An instance of Task with random priority and resources.
    """
    priority = random.randint(1, 10)
    resources = Resources(
        ram=random.randint(1, 32),
        cpu_cores=random.randint(1, 8),
        gpu_count=random.randint(0, 2)
    )
    content = f"Task Content {task_id}"
    logging.debug(f"Generated random task {task_id} with priority {priority} and resources {resources}")
    return Task(priority, task_id, resources, content)


def test_task_queue():
    """
    Tests adding tasks to the TaskQueue and fetching a task with specific available resources.
    """
    logging.info("Starting test_task_queue")
    queue = TaskQueue()

    for i in range(1000):
        task = generate_random_task(i)
        queue.add_task(task)
        logging.debug(f"Added {task}")

    available_resources = Resources(ram=16, cpu_cores=4, gpu_count=1)
    task = queue.get_task(available_resources)

    assert task is not None, "Expected to fetch a task, but got None."
    assert task.resources.is_sufficient(available_resources), \
        f"Task {task.id} resources {task.resources} are not sufficient for available resources {available_resources}."
    logging.info(f"Fetched Task ID: {task.id}, Priority: {task.priority}, Resources: {task.resources}")


def test_empty_queue():
    """
    Tests that no task is returned when the queue is empty.
    """
    logging.info("Starting test_empty_queue")
    queue = TaskQueue()
    available_resources = Resources(ram=8, cpu_cores=4, gpu_count=1)
    task = queue.get_task(available_resources)

    assert task is None, "Expected None when the queue is empty, but got a task."
    logging.info("test_empty_queue passed.")


def test_insufficient_resources():
    """
    Tests that no task is returned when the available resources are insufficient for all tasks.
    """
    logging.info("Starting test_insufficient_resources")
    queue = TaskQueue()

    task = Task(
        priority=10,
        id=99,
        resources=Resources(ram=64, cpu_cores=32, gpu_count=4),  # High requirements
        content="Impossible Task"
    )
    queue.add_task(task)
    logging.debug(f"Added high resource task {task}")

    available_resources = Resources(ram=1, cpu_cores=1, gpu_count=0)
    fetched_task = queue.get_task(available_resources)

    assert fetched_task is None, "Expected None when available resources are insufficient, but got a task."
    logging.info("test_insufficient_resources passed.")


def test_multiple_tasks_same_priority():
    """
    Tests that the task with the lowest ID is fetched when multiple tasks have the same priority.
    """
    logging.info("Starting test_multiple_tasks_same_priority")
    queue = TaskQueue()

    for i in range(10):
        task = Task(priority=5, id=i, resources=Resources(ram=2, cpu_cores=1, gpu_count=0), content=f"Task {i}")
        queue.add_task(task)
        logging.debug(f"Added task {task}")

    available_resources = Resources(ram=8, cpu_cores=4, gpu_count=1)
    task = queue.get_task(available_resources)

    assert task is not None and task.id == 0, \
        "Expected to fetch the task with the lowest ID among " \
        "those with the same priority, but fetched a different task."
    logging.info(f"test_multiple_tasks_same_priority passed, fetched Task ID: {task.id}.")


# Running the tests

if __name__ == '__main__':
    test_task_queue()
    test_empty_queue()
    test_insufficient_resources()
    test_multiple_tasks_same_priority()
