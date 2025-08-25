# 2️⃣ Create 'Bathrooms' feature
# df['Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

# # 3️⃣ Select features and target
# X = df[['GrLivArea', 'BedroomAbvGr', 'Bathrooms']]  # Features
# y = df['SalePrice']  # Target

# # 4️⃣ Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 5️⃣ Train Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 6️⃣ Predict on test set
# y_pred = model.predict(X_test)

# # 7️⃣ Evaluate model
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# # 8️⃣ Optional: show actual vs predicted
# comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print("\nFirst 10 predictions:\n", comparison.head(10))