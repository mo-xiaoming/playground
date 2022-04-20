package net.skelix.example003;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.ModelAndView;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@RestController
public class HelloController {
    @GetMapping("/hello")
    public String hello() {
        return "hello spring boot!";
    }

    @GetMapping("/book")
    public Book book() {
        var book = new Book();
        book.setAuthor("author1");
        book.setName("book1");
        book.setPrice(30f);
        book.setPublicationDate(new Date());
        return book;
    }
}
