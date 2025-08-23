class MyClass {
public:
  int i = 0;
  MyClass* next = nullptr;
};

int main() {
  MyClass* curr = new MyClass();
  while (curr->next == nullptr) {
    MyClass* myObject = new MyClass();
    myObject->next = curr;
    curr = myObject;
  }
}
