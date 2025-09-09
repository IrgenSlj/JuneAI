import java.util.*;

public class Array {
    
    public static void main(String[] args) {
        
        Collection<BankAccount> treeset = new TreeSet<>();

        BankAccount BA1 = new BankAccount("4534", 353, "Mike");
        BankAccount BA2 = new BankAccount("45234", 245432, "Kostas");
        BankAccount BA3 = new BankAccount("5644", 644765, "Kali");

        Collection<BankAccount> treeSet1 = new TreeSet<BankAccount>(new AccountCodeComparator);

        treeset.add(BA1);
        treeset.add(BA2);
        treeset.add(BA3);

        for (BankAccount account: treeset)
            System.out.println(account.getName() + ", " + account.getBalance() + ", " +  account.getCode());
    }
} 
