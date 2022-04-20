package net.skelix;

import javax.lang.model.type.NullType;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Stream;

class Worker implements Runnable {
    @Override
    public void run() {
        for (int i = 0; i < 1000; ++i)
            doWork();
    }

    private void doWork() {
        System.out.println("Hello world");
    }
}

class LengthComparator implements Comparator<String> {

    @Override
    public int compare(String s, String t1) {
        return Integer.compare(s.length(), t1.length());
    }
}

class Greeter {
    public void greet() {
        System.out.println("Hello world!");
    }
}

class ConcurrentGreeter extends Greeter {
    private void work() {
        System.out.println("Hello world from this");
    }

    public void greet() {
        new Thread(this::work).start();
        new Thread(super::greet).start();
    }
}

class Employee {
    private final String name;

    public Employee(String name) {
        this.name = name;
    }

    public String toString() {
        return this.name;
    }
}

public class Main {
    private static void workerExample() {
        Worker w = new Worker();
        new Thread(w).start();
    }

    private static void runnableExample() {
        Runnable sleeper = () -> {
            System.out.println("Zzz");
            try {
                Thread.sleep(1000L);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        new Thread(sleeper).start();
    }

    private static void sortArray() {
        var strings = new String[]{"Hello", "World!", "Hi"};
        Arrays.sort(strings, new LengthComparator());
        System.out.println(Arrays.toString(strings));

        Arrays.sort(strings, String::compareToIgnoreCase);
    }

    private static void sortArrayL() {
        var strings = new String[]{"Hello", "World!", "Hi"};
        Arrays.sort(strings, Comparator.comparing(String::length).reversed());
        System.out.println(Arrays.toString(strings));
    }

    private static void constructorNew() {
        var names = new String[]{"Jessica", "Lua", "Sprinkle"};
        var employees = Arrays.stream(names).peek(Logger.getGlobal()::info).map(Employee::new).toArray();
        System.out.println(Arrays.toString(employees));
    }

    private static void wordCounter() throws IOException {
        final var contents = Files.readString(Paths.get("src/alice.txt"));
        final var words = contents.split("\\R");
        int count = 0;
        for (final var w : words) {
            if (w.length() > 2)
                count++;
        }
        System.out.println(count);
    }

    private static void wordCounterL1() throws IOException {
        System.out.println(Files.lines(Paths.get("src/alice.txt")).parallel().filter(w -> w.length() > 2).count());
    }

    private static void wordCounterL() throws IOException {
        System.out.println(Stream.of(Files.readString(Paths.get("src/alice.txt")).split("\\R")).parallel().filter(w -> w.length() > 2).count());
    }

    private static void infiniteRandom(final int count) {
        Stream.generate(Math::random).limit(count).forEach(System.out::println);
    }

    private static void infiniteSequence(final int count) {
        Stream.iterate(BigInteger.ZERO, n -> n.add(BigInteger.ONE)).limit(count).forEach(System.out::println);
    }

    private static void peeking() {
        Stream.iterate(10, p -> p * 2)
                .peek(e -> System.out.println("Fetching " + e))
                .limit(20).forEach(System.out::println);
    }

    public static void main(String[] args) {
        constructorNew();
    }
}
