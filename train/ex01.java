import java.util.Scanner;

public class ex01 {
    public static void main(String[] args) {

        double width, height, area;

        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter the width: ");
        width = scanner.nextDouble(); 
        
        System.out.println("Enter the height: ");
        height = scanner.nextDouble(); 

        area = area(width, height);

        System.out.println("The area of the rectangle is: " + area + "m2");

        scanner.close();
    }
}