from dataclasses import dataclass
import dataclasses
from pprint import pprint
from typing import Optional

@dataclass
class Book:
    title: str
    author: str
    stars: Optional[int] = None  # can mark non-required fields with 'optional', but must supply a default (None works well with Optional)
    available: bool = True  # can set other reasonable defaults in class definition

if __name__ == '__main__':
    
    print("\n===== Basic instantiation =====\n")
    print("> book = Book(title='Tom Sawyer', author='Mark Twain')")
    book = Book(title='Tom Sawyer', author='Mark Twain')
    pprint(book)

    print("\n===== From a JSON/Dict =====\n")
    book_json = {'title': 'East of Eden', 'author': 'John Steinbeck'}
    print("> book_json = {'title': 'East of Eden', 'author': 'John Steinbeck'}")
    print("> book: Book = Book(**book_json)")

    # You can un-pack any dictionary with ** into keyword arguments to a function!
    book: Book = Book(**book_json)
    pprint(book)

    # similarly, you can **collect** un-caught key-word arguments to a function with **name, usually **kwargs

    def my_function(arg1, arg2='real_keyword_arg', **kwargs):
        print(kwargs)

    print("\n ==== Example ** arg collection: my_function(arg1, arg2='real_keyword_arg', **kwargs) =====\n")
    print("\n > my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d')\n")
    my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d')

    # combine with dictionary un-packing
    print("\n >my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d', **book_json)\n")
    my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d', **book_json)

    # can convert a dataclass to a dict with dataclasses.asdict()
    print("\n> my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d', **dataclasses.asdict(book))\n")
    my_function(arg1='a', arg2='b', c='c', any_other_named_arg_i_want='d', **dataclasses.asdict(book))
