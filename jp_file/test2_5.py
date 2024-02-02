class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value < 0:
            raise ValueError("Width cannot be negative")
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value < 0:
            raise ValueError("Height cannot be negative")
        self._height = value

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)

# 创建一个 Rectangle 对象
rectangle = Rectangle(5, 10)

# 访问属性和计算属性值
print(rectangle.width)  # 5
print(rectangle.height)  # 10
print(rectangle.area)  # 50
print(rectangle.perimeter)  # 30

# 修改属性值
rectangle.width = 7
rectangle.height = 12

print(rectangle.width)  # 7
print(rectangle.height)  # 12
print(rectangle.area)  # 84
print(rectangle.perimeter)  # 38


class MyClass:
    class_variable = "I am a class variable"

    @classmethod
    def my_class_method(cls):
        cls.class_variable=50
x=MyClass()
print(x.class_variable)
MyClass.my_class_method()
s=MyClass()
print(s.class_variable)
text = "   Hello, World!   "
print(text)
cleaned_text = text.strip()
print(cleaned_text)  # 输出："Hello, World!"
