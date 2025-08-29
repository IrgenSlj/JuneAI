import java.util.Scanner;

public class ex02 {
    public static void main(String[] args) {
        
        // SHOPPING CART PROGRAM
        
        Scanner scanner = new Scanner(System.in);

        String item;
        double price, total;
        int quantity;
        char currency = '$';

        System.out.println("What item to buy? ");
        item = scanner.nextLine();

        System.out.println("How many items would you like? ");
        quantity = scanner.nextInt();

        System.out.println("What is the price per item of " + item + " ? ");
        price = scanner.nextDouble();

        total = totalPrice(quantity, price);

        System.out.println("Total price is: " + total + currency);

        scanner.close();
    }
    public static double totalPrice(int quantity, double price) {
        return quantity * price;
    }
}