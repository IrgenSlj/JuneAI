import java.util.Scanner;

public class ex03 {
    public static void main(String[] args) {
        int age;
        String name;

        Scanner scanner = new Scanner(System.in);

        System.out.print("What's your age? ");
        age = scanner.nextInt();

        System.out.print("\nWhat's your name? ");
        name = scanner.nextLine();

        if (name.isEmpty()) {
            System.out.print("\nError name! Try again: ");
            name = scanner.nextLine();
        } else {
            System.out.println("Hello " + name + " !");
        }

        printAge(name, age);

        scanner.close();

    }
    public static void printAge(String name, int age) {

        if (age >= 18) {
            System.out.println(name + " ,you are an adult");
        } else if (age >= 65) {
            System.out.println(name + " ,you are a senior");
        } else if (age < 0) {
            System.out.println(name + " ,you haven't been born yet");
        } else if (age < 18) {
            System.out.println(name + " ,you are underaged");
        } else {
            System.out.println(name + " ,new born!");
        }
    }
    
}
