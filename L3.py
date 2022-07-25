# -*- coding: utf-8 -*-
from enum import Enum

class UserType(Enum):  
    ADMIN = "Admin"
    READER = "Reader"
    WRITER = "Writer"
    DEBUG = "Debug"


class WebPage():
    
    VALUE = 5

    def __init__(self, www_address):
        self.www_address = www_address
        self.ip_address = None
        self.__content = []
        self.hyperlinks = {}
        self.pictures = {}
    
    def __str__(self):
        return self.__content
        
    def get_content(self):
        content = ""
        for line in self.__content:
            content += line
        return content
    
    def lines(self, no_of_lines):
        for line in self.__content[:no_of_lines]:
            yield line
    
    def grep_content(self, word):
        pass
    
    def get_hyperlink_address(self, hyperlink_name):
        pass
    
    def get_file_content(self, source_file):
        with open(source_file) as file:
            self.__content += file.readlines()
            
    def remove_blank_lines(self):
        pattern = '\n'
        for line in self.__content:
            if line == pattern:
                self.__content.remove(pattern)
    
    
class SpecialWebPage(WebPage):
    number_of_webpages = 0

    
    def __init__(self, www_address, language):
        super().__init__(www_address)    
        self.language = language
        self.special_characteristics = None
        self._check_www_address()
        SpecialWebPage.counter()
        
        
    def counter():
        SpecialWebPage.number_of_webpages += 1
        print(SpecialWebPage.number_of_webpages)
     
    def _check_www_address(self):
        if type(self.www_address) is str:
            print("OK!")
        else:
            raise TypeError
     
    def translate(self, output_language):
        pass


class TextSpecialWebPage(SpecialWebPage):
    
    def __init__(self, www_address, language, source_file):
        super().__init__(www_address, language)
        self.get_file_content(source_file)