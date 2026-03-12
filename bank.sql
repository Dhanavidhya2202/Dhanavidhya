create database Bank_Management_System;
use Bank_Management_System;
create table Customer(Customer_id int primary key,First_name varchar(20),Last_name varchar(20),Gender varchar(10),
DoB date,Phone varchar(20),Email varchar(50),City varchar(10),State varchar(10),Branch varchar(20),Bank_Name varchar(20),
Account_Type varchar(20),Account_no varchar(30),IFSC_code varchar(20),Transaction_Type varchar(20),Loan varchar(10),Loan_Type varchar(20),
Payment_date date,ATM_Transaction_Type varchar(20),KYC_id int unique); 
select * from Customer;
insert into Customer values
(1,'Arun','Kumar','Male','1995-05-14','9876543210','arun1@gmail.com','Chennai','TamilNadu','T Nagar','SBI','Savings','SB1001','SBIN0001','Deposit','No',NULL,'2025-01-01','Cash Withdrawal',101),
(2,'Priya','Sharma','Female','1998-08-21','9876543211','priya2@gmail.com','Mumbai','Maharashtra','Andheri','HDFC','Current','HD2001','HDFC0002','Withdrawal','Yes','Home Loan','2025-01-02','Balance Enquiry',102),
(3,'Rahul','Verma','Male','1992-11-10','9876543212','rahul3@gmail.com','Delhi','Delhi','KarolBagh','ICICI','Savings','IC3001','ICIC0003','Deposit','Yes','Car Loan','2025-01-03','Mini Statement',103),
(4,'Sneha','Reddy','Female','1996-02-18','9876543213','sneha4@gmail.com','Hyderabad','Telangana','Madhapur','Axis','Savings','AX4001','UTIB0004','Transfer','No',NULL,'2025-01-04','Cash Deposit',104),
(5,'Vikram','Singh','Male','1993-07-09','9876543214','vikram5@gmail.com','Bangalore','Karnataka','MG Road','SBI','Current','SB1002','SBIN0005','Withdrawal','Yes','Personal Loan','2025-01-05','Cash Withdrawal',105),
(6,'Anjali','Patel','Female','1997-03-12','9876543215','anjali6@gmail.com','Ahmedabad','Gujarat','Navrangpura','HDFC','Savings','HD2002','HDFC0006','Deposit','No',NULL,'2025-01-06','Balance Enquiry',106),
(7,'Karthik','Nair','Male','1991-06-25','9876543216','karthik7@gmail.com','Kochi','Kerala','Edappally','ICICI','Savings','IC3002','ICIC0007','Transfer','Yes','Education Loan','2025-01-07','Mini Statement',107),
(8,'Meena','Iyer','Female','1999-09-30','9876543217','meena8@gmail.com','Chennai','TamilNadu','Adyar','Axis','Savings','AX4002','UTIB0008','Deposit','No',NULL,'2025-01-08','Cash Deposit',108),
(9,'Rohit','Gupta','Male','1994-04-11','9876543218','rohit9@gmail.com','Jaipur','Rajasthan','MI Road','SBI','Savings','SB1003','SBIN0009','Withdrawal','Yes','Home Loan','2025-01-09','Cash Withdrawal',109),
(10,'Divya','Menon','Female','1995-12-22','9876543219','divya10@gmail.com','Pune','Maharashtra','ShivajiNagar','HDFC','Current','HD2003','HDFC0010','Deposit','No',NULL,'2025-01-10','Balance Enquiry',110),
(11,'Ajay','Das','Male','1990-01-14','9876543220','ajay11@gmail.com','Kolkata','WestBengal','SaltLake','ICICI','Savings','IC3003','ICIC0011','Transfer','Yes','Car Loan','2025-01-11','Mini Statement',111),
(12,'Lakshmi','Rao','Female','1993-10-18','9876543221','lakshmi12@gmail.com','Visakhapatnam','AndhraPradesh','Dwaraka','Axis','Savings','AX4003','UTIB0012','Deposit','No',NULL,'2025-01-12','Cash Deposit',112),
(13,'Manoj','Yadav','Male','1996-07-07','9876543222','manoj13@gmail.com','Lucknow','UttarPradesh','Hazratganj','SBI','Savings','SB1004','SBIN0013','Withdrawal','Yes','Personal Loan','2025-01-13','Cash Withdrawal',113),
(14,'Neha','Kapoor','Female','1998-03-03','9876543223','neha14@gmail.com','Chandigarh','Punjab','Sector17','HDFC','Current','HD2004','HDFC0014','Deposit','No',NULL,'2025-01-14','Balance Enquiry',114),
(15,'Suresh','Babu','Male','1992-12-19','9876543224','suresh15@gmail.com','Madurai','TamilNadu','AnnaNagar','ICICI','Savings','IC3004','ICIC0015','Transfer','Yes','Home Loan','2025-01-15','Mini Statement',115),
(16,'Pooja','Joshi','Female','1997-06-28','9876543225','pooja16@gmail.com','Nagpur','Maharashtra','Sitabuldi','Axis','Savings','AX4004','UTIB0016','Deposit','No',NULL,'2025-01-16','Cash Deposit',116),
(17,'Aravind','Raj','Male','1994-09-05','9876543226','aravind17@gmail.com','Coimbatore','TamilNadu','RS Puram','SBI','Current','SB1005','SBIN0017','Withdrawal','Yes','Car Loan','2025-01-17','Cash Withdrawal',117),
(18,'Deepa','Shah','Female','1995-11-23','9876543227','deepa18@gmail.com','Surat','Gujarat','Varachha','HDFC','Savings','HD2005','HDFC0018','Deposit','No',NULL,'2025-01-18','Balance Enquiry',118),
(19,'Naveen','Chowdhury','Male','1991-04-17','9876543228','naveen19@gmail.com','Patna','Bihar','Kankarbagh','ICICI','Savings','IC3005','ICIC0019','Transfer','Yes','Education Loan','2025-01-19','Mini Statement',119),
(20,'Shalini','Mishra','Female','1999-02-01','9876543229','shalini20@gmail.com','Bhopal','MadhyaPradesh','MP Nagar','Axis','Savings','AX4005','UTIB0020','Deposit','No',NULL,'2025-01-20','Cash Deposit',120),
(21,'Kiran','Patil','Male','1993-08-08','9876543230','kiran21@gmail.com','Mumbai','Maharashtra','Dadar','SBI','Savings','SB1006','SBIN0021','Deposit','No',NULL,'2025-01-21','Cash Withdrawal',121),
(22,'Riya','Agarwal','Female','1996-01-09','9876543231','riya22@gmail.com','Delhi','Delhi','Rohini','HDFC','Current','HD2006','HDFC0022','Withdrawal','Yes','Home Loan','2025-01-22','Balance Enquiry',122),
(23,'Varun','Malhotra','Male','1992-05-20','9876543232','varun23@gmail.com','Noida','UttarPradesh','Sector62','ICICI','Savings','IC3006','ICIC0023','Deposit','Yes','Car Loan','2025-01-23','Mini Statement',123),
(24,'Keerthi','Raman','Female','1997-10-12','9876543233','keerthi24@gmail.com','Chennai','TamilNadu','Velachery','Axis','Savings','AX4006','UTIB0024','Transfer','No',NULL,'2025-01-24','Cash Deposit',124),
(25,'Gokul','Krishnan','Male','1994-06-30','9876543234','gokul25@gmail.com','Trichy','TamilNadu','Srirangam','SBI','Savings','SB1007','SBIN0025','Withdrawal','Yes','Personal Loan','2025-01-25','Cash Withdrawal',125),
(26,'Aisha','Khan','Female','1995-04-15','9876543235','aisha26@gmail.com','Hyderabad','Telangana','BanjaraHills','HDFC','Savings','HD2007','HDFC0026','Deposit','No',NULL,'2025-01-26','Balance Enquiry',126),
(27,'Imran','Ali','Male','1993-09-09','9876543236','imran27@gmail.com','Lucknow','UttarPradesh','Alambagh','ICICI','Savings','IC3007','ICIC0027','Transfer','Yes','Home Loan','2025-01-27','Mini Statement',127),
(28,'Sanjana','Gupta','Female','1998-12-02','9876543237','sanjana28@gmail.com','Indore','MadhyaPradesh','VijayNagar','Axis','Savings','AX4007','UTIB0028','Deposit','No',NULL,'2025-01-28','Cash Deposit',128),
(29,'Harish','Nandan','Male','1991-07-21','9876543238','harish29@gmail.com','Mysore','Karnataka','VV Mohalla','SBI','Current','SB1008','SBIN0029','Withdrawal','Yes','Car Loan','2025-01-29','Cash Withdrawal',129),
(30,'Bhavya','Shetty','Female','1996-03-14','9876543239','bhavya30@gmail.com','Mangalore','Karnataka','Kadri','HDFC','Savings','HD2008','HDFC0030','Deposit','No',NULL,'2025-01-30','Balance Enquiry',130),
(31,'Tarun','Mehta','Male','1994-02-11','9876543240','tarun31@gmail.com','Jaipur','Rajasthan','MalviyaNagar','ICICI','Savings','IC3008','ICIC0031','Transfer','Yes','Education Loan','2025-02-01','Mini Statement',131),
(32,'Nisha','Saxena','Female','1997-06-19','9876543241','nisha32@gmail.com','Delhi','Delhi','LajpatNagar','Axis','Savings','AX4008','UTIB0032','Deposit','No',NULL,'2025-02-02','Cash Deposit',132),
(33,'Vignesh','Kumar','Male','1995-10-25','9876543242','vignesh33@gmail.com','Chennai','TamilNadu','Porur','SBI','Savings','SB1009','SBIN0033','Withdrawal','Yes','Home Loan','2025-02-03','Cash Withdrawal',133),
(34,'Aparna','R','Female','1998-01-16','9876543243','aparna34@gmail.com','Coimbatore','TamilNadu','SaibabaColony','HDFC','Current','HD2009','HDFC0034','Deposit','No',NULL,'2025-02-04','Balance Enquiry',134),
(35,'Pradeep','Naik','Male','1992-08-28','9876543244','pradeep35@gmail.com','Goa','Goa','Panaji','ICICI','Savings','IC3009','ICIC0035','Transfer','Yes','Personal Loan','2025-02-05','Mini Statement',135),
(36,'Kavitha','S','Female','1993-11-11','9876543245','kavitha36@gmail.com','Salem','TamilNadu','Fairlands','Axis','Savings','AX4010','UTIB0036','Deposit','No',NULL,'2025-02-06','Cash Deposit',136),
(37,'Ramesh','Pillai','Male','1991-05-05','9876543246','ramesh37@gmail.com','Thiruvananthapuram','Kerala','Kowdiar','SBI','Savings','SB1010','SBIN0037','Withdrawal','Yes','Car Loan','2025-02-07','Cash Withdrawal',137),
(38,'Shruti','Desai','Female','1996-09-17','9876543247','shruti38@gmail.com','Vadodara','Gujarat','Alkapuri','HDFC','Savings','HD2010','HDFC0038','Deposit','No',NULL,'2025-02-08','Balance Enquiry',138),
(39,'Mahesh','I','Male','1994-04-04','9876543248','mahesh39@gmail.com','Bangalore','Karnataka','Whitefield','ICICI','Savings','IC3010','ICIC0039','Transfer','Yes','Home Loan','2025-02-09','Mini Statement',139),
(40,'Rekha','T','Female','1997-07-07','9876543249','rekha40@gmail.com','Chennai','TamilNadu','Tambaram','Axis','Savings','AX4011','UTIB0040','Deposit','No',NULL,'2025-02-10','Cash Deposit',140),
(41,'Sathish','K','Male','1993-03-03','9876543250','sathish41@gmail.com','Erode','TamilNadu','Perundurai','SBI','Savings','SB1011','SBIN0041','Withdrawal','Yes','Personal Loan','2025-02-11','Cash Withdrawal',141),
(42,'Monica','L','Female','1998-12-12','9876543251','monica42@gmail.com','Pune','Maharashtra','Hinjewadi','HDFC','Savings','HD2011','HDFC0042','Deposit','No',NULL,'2025-02-12','Balance Enquiry',142),
(43,'Ganesh','M','Male','1992-06-06','9876543252','ganesh43@gmail.com','Madurai','TamilNadu','KK Nagar','ICICI','Savings','IC3011','ICIC0043','Transfer','Yes','Education Loan','2025-02-13','Mini Statement',143),
(44,'Swathi','N','Female','1995-09-09','9876543253','swathi44@gmail.com','Hyderabad','Telangana','Secunderabad','Axis','Savings','AX4012','UTIB0044','Deposit','No',NULL,'2025-02-14','Cash Deposit',144),
(45,'Dinesh','R','Male','1994-01-01','9876543254','dinesh45@gmail.com','Chennai','TamilNadu','AnnaNagar','SBI','Current','SB1012','SBIN0045','Withdrawal','Yes','Home Loan','2025-02-15','Cash Withdrawal',145),
(46,'Preethi','V','Female','1997-05-15','9876543255','preethi46@gmail.com','Coimbatore','TamilNadu','Gandhipuram','HDFC','Savings','HD2012','HDFC0046','Deposit','No',NULL,'2025-02-16','Balance Enquiry',146),
(47,'Nitin','S','Male','1991-11-11','9876543256','nitin47@gmail.com','Nagpur','Maharashtra','Dharampeth','ICICI','Savings','IC3012','ICIC0047','Transfer','Yes','Car Loan','2025-02-17','Mini Statement',147),
(48,'Harini','P','Female','1996-08-08','9876543257','harini48@gmail.com','Chennai','TamilNadu','Chromepet','Axis','Savings','AX4013','UTIB0048','Deposit','No',NULL,'2025-02-18','Cash Deposit',148),
(49,'Arjun','L','Male','1993-02-20','9876543258','arjun49@gmail.com','Bangalore','Karnataka','Indiranagar','SBI','Savings','SB1013','SBIN0049','Withdrawal','Yes','Personal Loan','2025-02-19','Cash Withdrawal',149),
(50,'Keerthana','M','Female','1999-10-10','9876543259','keerthana50@gmail.com','Salem','TamilNadu','Ammapet','HDFC','Savings','HD2013','HDFC0050','Deposit','No',NULL,'2025-02-20','Balance Enquiry',150),
(51,'Vasanth','K','Male','1992-07-07','9876543260','vasanth51@gmail.com','Trichy','TamilNadu','ThillaiNagar','ICICI','Savings','IC3013','ICIC0051','Transfer','Yes','Home Loan','2025-02-21','Mini Statement',151),
(52,'Sangeetha','R','Female','1995-06-06','9876543261','sangeetha52@gmail.com','Pondicherry','Puducherry','WhiteTown','Axis','Savings','AX4014','UTIB0052','Deposit','No',NULL,'2025-02-22','Cash Deposit',152),
(53,'Lokesh','B','Male','1994-03-13','9876543262','lokesh53@gmail.com','Mumbai','Maharashtra','Borivali','SBI','Savings','SB1014','SBIN0053','Withdrawal','Yes','Car Loan','2025-02-23','Cash Withdrawal',153),
(54,'Anu','C','Female','1997-01-21','9876543263','anu54@gmail.com','Delhi','Delhi','Dwarka','HDFC','Current','HD2014','HDFC0054','Deposit','No',NULL,'2025-02-24','Balance Enquiry',154),
(55,'Rajesh','T','Male','1991-12-30','9876543264','rajesh55@gmail.com','Kolkata','WestBengal','Howrah','ICICI','Savings','IC3014','ICIC0055','Transfer','Yes','Education Loan','2025-02-25','Mini Statement',155),
(56,'Pavithra','D','Female','1998-02-02','9876543265','pavithra56@gmail.com','Madurai','TamilNadu','AnnaBusStand','Axis','Savings','AX4015','UTIB0056','Deposit','No',NULL,'2025-02-26','Cash Deposit',156),
(57,'Arav','G','Male','1993-09-19','9876543266','arav57@gmail.com','Surat','Gujarat','Adajan','SBI','Savings','SB1015','SBIN0057','Withdrawal','Yes','Personal Loan','2025-02-27','Cash Withdrawal',157),
(58,'Kavya','H','Female','1996-04-24','9876543267','kavya58@gmail.com','Hyderabad','Telangana','Gachibowli','HDFC','Savings','HD2015','HDFC0058','Deposit','No',NULL,'2025-02-28','Balance Enquiry',158),
(59,'Madhan','J','Male','1992-05-15','9876543268','madhan59@gmail.com','Chennai','TamilNadu','Perambur','ICICI','Savings','IC3015','ICIC0059','Transfer','Yes','Home Loan','2025-03-01','Mini Statement',159),
(60,'Sowmya','K','Female','1999-08-08','9876543269','sowmya60@gmail.com','Bangalore','Karnataka','Jayanagar','Axis','Savings','AX4016','UTIB0060','Deposit','No',NULL,'2025-03-02','Cash Deposit',160);
alter table Customer modify State varchar(20);
alter table Customer modify City varchar(20);

-- The Chennai branch manager wants to see all customers from TamilNadu.--
select * from Customer where State="Tamilnadu";

-- HR wants the list of all female customers.--
select * from Customer where Gender="Female";

-- The branch wants to identify all Savings account holders.--
select * from Customer where Account_Type="Savings";

-- Loan department wants to see customers who have taken loans.--
select * from Customer where Loan="Yes";

-- A customer support executive wants to search customer details using Account Number.--
select * from Customer where Account_no="SB1001";

-- The bank wants to list customers from a specific city.--
select * from Customer where City="Chennai";

-- The bank wants to know the total number of customers.--
select count(*) as total_customer from Customer;

-- HR wants to know male vs female customer count.--
select Gender, count(*) as total_customer from Customer group by Gender;

-- The bank wants to know which bank (SBI/HDFC/ICICI) has more customers.--
select Bank_Name,count(*) as total_customer from Customer where Bank_Name in ('SBI','HDFC','ICICI') group by Bank_Name order by total_customer desc;

-- The head office wants a report of customer count by city.--
 select city,count(*) as total_customer from Customer group by city order by total_customer desc;
 
 -- The bank wants to know how many Current account holders exist.--
 select count(*) as total_current_account_holders from Customer where Account_Type="Current";
 
 -- The bank wants to know total customers per account type.--
 select Account_Type,count(*) as total_customer from Customer group by Account_Type order by total_customer desc;
 
 -- The admin wants states having more than 5 customers.--
 select State,count(*) as total_customer from Customer group by State having count(*)>5 order by total_customer desc;
 
 -- The bank wants all customer names displayed in uppercase for reporting.--
 select upper(First_name) as First_name,upper(Last_name) as Last_name from Customer;
 
 -- The system needs to generate full names by combining first and last names.--
 select concat(First_name,' ',Last_name) as full_name from Customer;
 
 -- The IT team wants to check the length of email IDs.--
 select Email,Length(Email) as Length from Customer;
 
 -- The manager wants to extract the first 3 letters of city names.--
 select City, substring(City,1,3) as city_code from Customer;
 
 -- The bank wants to replace “TamilNadu” with “TN” in reports.--
 select replace(State,"TamilNadu","TN") as State_name from Customer;
 
 -- The admin wants to display emails in lowercase format.--
 select lower(Email) as Email from Customer;
 
 -- The bank wants to check customers whose names start with “A”.--
select * from Customer where First_name like "A%";
 
 -- The bank wants to find customers whose email contains “gmail”.--
 select * from Customer where Email like "%gmail%" ;

 -- The system wants to check phone numbers.--
 select Phone from Customer;
 
 -- The manager wants to list customers above 30 years.--
 select First_name from Customer where DoB <= date_sub(curdate(), interval 30 year);
 
 -- The bank wants customers born between 1995 and 1999.--
 select * from Customer where year(DoB) between '1995' and '1999';
 
 -- The finance team wants to see customers who made payments last 2 month.--
 select * from Customer where month(Payment_date) between '02' and '03'; 
 
 -- The management wants to find customers from the state that has the highest number of customers.--
 select State from Customer group by State order by count(*) desc limit 2;
 
 -- The branch wants to find customers from the same city as a specific customer.--
 select * from Customer as c1 inner join Customer as c2 on c1.City=c2.City where c2.Customer_id=5;
 
 -- The admin wants to list customers belonging to multiple branches.--
 select distinct c1.First_name,c1.Branch from Customer as c1 join Customer as c2 on c1.Customer_id=c2.Customer_id and c1.Branch=c2.Branch;
 
 -- The manager wants customers from Chennai or Bangalore.--
select First_name,City from Customer where City='Chennai' or City='Bangalore';

 -- The admin wants customers from selected states.--
select * from Customer where State in ('Karnataka');
