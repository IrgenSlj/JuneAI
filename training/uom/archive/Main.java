
import java.util.ArrayList;
import java.util.HashMap;

public class Main {
    
    public static void main(String[] args) {

        ArrayList<String> johnsBooks = new ArrayList<>();
        johnsBooks.add("Apollo 13");
        johnsBooks.add("Alchemist");
        johnsBooks.add("Kalamia");

        ArrayList<String> marryBooks = new ArrayList<>();
        marryBooks.add("Hearts");
        marryBooks.add("Karma");
        marryBooks.add("Sirma");

        HashMap<String, ArrayList<String>> library = 
            new HashMap<>();

        library.put("John", johnsBooks);
        library.put("Marry", marryBooks);

        for (String member: library.keySet()) {
            System.out.println(member + " has borrowed: ");
            for (String book: library.get(member)) 
                System.out.println(book);
            
            System.out.println();
        }

    }
    
}
